# coding: utf-8

from __future__ import unicode_literals

import collections
import json
import logging

from rasa_nlu.data_router import DataRouter

from seabattle import game


log = logging.getLogger(__name__)
router = DataRouter('mldata/')
MESSAGE_TEMPLATES = {
    'miss': 'Мимо. Я хожу %(shot)s',
    'hit': 'Ты попала',
    'kill': 'Корабль утонул',
    'newgame': 'Инициализирована новая игра c %(opponent)s',
    'shot': 'Я хожу %(shot)s',
    'defeat': 'Я проиграла',
    'victory': 'Ура, победа!',
    'need_init': 'Пожалуйста, инициализируй новую игру и укажи соперника',
    'dontunderstand': 'Не поняла. Повтори последний ход'
}
TTS_TEMPLATES = {
    'newgame': 'Инициализирована новая игра с - - %(opponent)s',
    'miss': 'Мимо - Я хожу - %(tts_shot)s',
    'shot': 'Я хожу - %(tts_shot)s',
}
DMResponse = collections.namedtuple('DMResponse', ['key', 'text', 'tts', 'end_session'])


def _get_entity(entities, entity_type):

    try:
        return next(e['value'] for e in entities if e['entity'] == entity_type)
    except StopIteration:
        return None


def _shot_to_tts(shot):
    return shot.replace(', ', ' - - - - ')


class DialogManager(object):
    def __init__(self, session_obj):
        self.session = session_obj
        self.game = session_obj['game']
        self.opponent = session_obj['opponent']
        self.last = session_obj['last']

    def _get_dmresponse(self, key, text, tts=None, end_session=False, with_opponent=False):
        if with_opponent and not text.lower().startswith(self.opponent.lower()):
            text = '%s, %s' % (self.opponent, text)
            if tts:
                tts = '%s - - %s' % (self.opponent, tts)
        return DMResponse(key, text, tts, end_session)

    def _get_shot_miss_dmresponse(self, key, shot, with_opponent=False):
        response_dict = {
            'shot': shot,
            'tts_shot': _shot_to_tts(shot),
        }
        return self._get_dmresponse(
            key,
            MESSAGE_TEMPLATES[key] % response_dict,
            TTS_TEMPLATES[key] % response_dict,
            with_opponent=with_opponent
        )

    def _get_dmresponse_by_key(self, key, end_session=False, with_opponent=False):
        return self._get_dmresponse(
            key,
            MESSAGE_TEMPLATES[key],
            end_session=end_session,
            with_opponent=with_opponent
        )

    def _handle_newgame(self, message, entities):
        self.game = game.Game()
        self.game.reset_last_shot()
        self.session['game'] = self.game
        self.game.start_new_game(numbers=True)
        if entities:
            self.opponent = _get_entity(entities, 'opponent_entity')
        else:
            self.opponent = 'Алиса'

        self.session['opponent'] = self.opponent
        response_dict = {'opponent': self.opponent}
        return self._get_dmresponse(
            'newgame',
            MESSAGE_TEMPLATES['newgame'] % response_dict,
            TTS_TEMPLATES['newgame'] % response_dict,
        )

    def _handle_letsstart(self, message, entities):
        if self.game is None:
            return self._get_dmresponse_by_key('need_init')
        self.game.reset_last_shot()
        shot = self.game.do_shot()
        return self._get_shot_miss_dmresponse('shot', shot, with_opponent=True)

    def _handle_miss(self, message, entities):
        if self.game is None:
            return self._get_dmresponse_by_key('need_init')

        self.game.handle_enemy_reply('miss')
        if not entities:
            return self._get_dmresponse_by_key('dontunderstand')
        enemy_shot = _get_entity(entities, 'hit_entity')
        try:
            enemy_position = self.game.convert_to_position(enemy_shot)
            answer = self.game.handle_enemy_shot(enemy_position)
        except ValueError:
            return self._get_dmresponse_by_key('dontunderstand')
        if answer == 'miss':
            shot = self.game.do_shot()
            return self._get_shot_miss_dmresponse('miss', shot)
        return self._get_dmresponse(
            answer,
            MESSAGE_TEMPLATES[answer],
        )

    def _handle_hit(self, message, entities):
        if self.game is None:
            return self._get_dmresponse_by_key('need_init')

        self.game.handle_enemy_reply('hit')
        shot = self.game.do_shot()
        return self._get_shot_miss_dmresponse('shot', shot)

    def _handle_kill(self, message, entities):
        if self.game is None:
            return self._get_dmresponse_by_key('need_init')

        self.game.handle_enemy_reply('kill')
        shot = self.game.do_shot()
        if self.game.is_victory():
            return self._get_dmresponse_by_key('victory')
        else:
            return self._get_shot_miss_dmresponse('shot', shot)

    def _handle_dontunderstand(self, message, entities):
        if self.game is None:
            return self._get_dmresponse_by_key('need_init')
        self.game.reset_last_shot()

        if self.last.key in ['miss', 'shot']:
            shot = self.game.repeat()
            return self._get_shot_miss_dmresponse(self.last.key, shot, with_opponent=True)
        return self._get_dmresponse(self.last.key, self.last.text, with_opponent=True)

    def _handle_victory(self, message, entities):
        self.session['game'] = self.game = None
        return self._get_dmresponse_by_key('defeat', True)

    def _handle_defeat(self, message, entities):
        self.session['game'] = self.game = None
        return self._get_dmresponse_by_key('victory', True)

    def _update_session(self, dmresponse):
        self.session['last'] = self.last = dmresponse

    def handle_message(self, message):
        data = router.extract({'q': message})
        router_response = router.parse(data)
        log.info('Router response %s', json.dumps(router_response, indent=2))

        if router_response['intent']['confidence'] < 0.8:
            dmresponse = self._get_dmresponse_by_key('dontunderstand')
            return dmresponse

        intent_name = router_response['intent']['name']
        entities = router_response['entities']
        handler_method = getattr(self, '_handle_' + intent_name)
        dmresponse = handler_method(message, entities)
        if dmresponse.key != 'dontunderstand':
            # сохраняем только последний осмысленный ответ в сессии не затыкались после нескольких повтори
            self._update_session(dmresponse)

        if self.session.get('game') is not None:
            log.info('My field:')
            self.session['game'].print_field()
            log.info('Enemy field:')
            self.session['game'].print_enemy_field()

        return dmresponse

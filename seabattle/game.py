# coding: utf-8

from __future__ import unicode_literals

import random
import re

from transliterate import translit

EMPTY = 0
SHIP = 1
HIT = 2
BLOCKED = 3
MISS = 4


class Game(object):
    position_patterns = [re.compile('^([a-zа-я]+)(\d+)$', re.UNICODE),  # a1
                         re.compile('^([a-zа-я]+)\s+(\w+)$', re.UNICODE),  # a 1; a один
                         re.compile('^(\w+)\s+(\w+)$', re.UNICODE),  # a 1; a один; 7 10
                         re.compile('^(\w+)$', re.UNICODE)  # 1; один
                         ]

    str_letters = ['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'к']
    str_numbers = ['один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять', 'десять']

    letters_mapping = {'the': 'з',
                       'за': 'з',
                       'уже': 'ж'}

    ships = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]

    def __init__(self):
        self.size = 0
        self.field = []
        self.enemy_field = []

        self.ships_count = 0
        self.enemy_ships_count = 0

        self.last_shot_position = None
        self.last_enemy_shot_position = None

    def start_new_game(self, size=10, field=None):
        assert(size <= 10)
        assert(len(field) == size ** 2 if field is not None else True)

        self.size = size

        if field is None:
            self.generate_field()
        else:
            self.field = field

        self.enemy_field = [EMPTY] * self.size ** 2

        self.ships_count = self.enemy_ships_count = len(self.ships)

        self.last_shot_position = None
        self.last_enemy_shot_position = None

    def generate_field(self):
        self.field = [0] * self.size ** 2

        for length in self.ships:
            self.place_ship(length)

        for i in range(0, len(self.field)):
            if self.field[i] == BLOCKED:
                self.field[i] = EMPTY

    def print_field(self):
        mapping = [' ', '0', 'x']

        print '---'
        for y in range(self.size):
            print '|%s|' % ''.join(mapping[x] for x in self.field[y * self.size: y * self.size + self.size])
        print '---'

    def place_ship(self, length):
        def _try_to_place():
            x = random.randint(1, self.size)
            y = random.randint(1, self.size)
            direction = random.choice([1, self.size])

            index = self.calc_index((x, y))
            values = self.field[index:None if direction != 1 else index + self.size - index % self.size:direction][:length]

            if len(values) < length or any(values):
                return False

            for i in range(0, length):
                current_index = index + direction * i

                for j in [0, 1, -1]:
                    if (current_index % self.size in (0, self.size - 1)
                            and (current_index + j) % self.size in (0, self.size - 1)):
                        continue

                    for k in [0, self.size, -self.size]:
                        neighbour_index = current_index + k + j

                        if (neighbour_index < 0
                                or neighbour_index >= len(self.field)
                                or self.field[neighbour_index] == SHIP):
                            continue

                        self.field[neighbour_index] = BLOCKED

                self.field[current_index] = SHIP

            return True

        while not _try_to_place():
            pass

    def handle_enemy_shot(self, position):
        index = self.calc_index(position)

        if self.field[index] == SHIP:
            self.field[index] = HIT

            if self.is_dead_ship(index):
                self.ships_count -= 1
                return 'kill'
            else:
                return 'hit'
        else:
            return 'miss'

    def is_dead_ship(self, last_index):
        x, y = self.calc_position(last_index)
        x -= 1
        y -= 1

        def _line_is_dead(line, index):
            def _tail_is_dead(tail):
                for i in tail:
                    if i == HIT:
                        continue
                    elif i == SHIP:
                        return False
                    else:
                        return True
                return True

            return _tail_is_dead(line[index:]) and _tail_is_dead(line[index::-1])

        return (
            _line_is_dead(self.field[x::self.size], y) and
            _line_is_dead(self.field[y * self.size:(y + 1) * self.size], x)
        )

    def is_end_game(self):
        return self.is_victory() or self.is_defeat()

    def is_victory(self):
        return self.enemy_ships_count < 1

    def is_defeat(self):
        return self.ships_count < 1

    def do_shot(self):
        index = random.choice([i for i, v in enumerate(self.enemy_field) if v == EMPTY])

        self.last_shot_position = self.calc_position(index)
        return self.convert_from_position(self.last_shot_position)

    def repeat(self):
        return self.convert_from_position(self.last_shot_position, numbers=True)

    def handle_enemy_reply(self, message):
        if self.last_shot_position is None:
            return

        index = self.calc_index(self.last_shot_position)

        if message in ['hit', 'kill']:
            self.enemy_field[index] = SHIP

            if message == 'kill':
                self.enemy_ships_count -= 1

        elif message == 'miss':
            self.enemy_field[index] = MISS

    def calc_index(self, position):
        x, y = position

        if x > self.size or y > self.size:
            raise ValueError('Wrong position: %s %s' % (x, y))

        return (y - 1) * self.size + x - 1

    def calc_position(self, index):
        y = index / self.size + 1
        x = index % self.size + 1

        return x, y

    def convert_to_position(self, position):
        position = position.lower()
        for pattern in self.position_patterns:
            match = pattern.match(position)

            if match is not None:
                break
        else:
            raise ValueError('Can\'t parse entire position: %s' % position)

        bits = match.groups()

        if len(bits) == 1:
            bits = ('а', bits[0])

        def _try_letter(bit):
            # проверяем особые случаи неправильного распознования STT
            bit = self.letters_mapping.get(bit, bit)

            # преобразуем в кириллицу
            bit = translit(bit, 'ru')

            try:
                return self.str_letters.index(bit) + 1
            except ValueError:
                raise

        def _try_number(bit):
            if bit.isdigit():
                return int(bit)
            else:
                try:
                    return self.str_numbers.index(bit) + 1
                except ValueError:
                    raise

        x = bits[0].strip()
        try:
            x = _try_letter(x)
        except ValueError:
            try:
                x = _try_number(x)
            except ValueError:
                raise ValueError('Can\'t parse X point: %s' % x)

        y = bits[1].strip()
        try:
            y = _try_number(y)
        except ValueError:
            raise ValueError('Can\'t parse Y point: %s' % y)

        return x, y

    def convert_from_position(self, position, numbers=False):
        if numbers:
            x = position[0]
        else:
            x = self.str_letters[position[0] - 1]

        y = position[1]

        return '%s %s' % (x, y)

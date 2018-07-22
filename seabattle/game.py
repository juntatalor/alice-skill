# coding: utf-8

from __future__ import unicode_literals

import random
import re
import logging

from transliterate import translit

EMPTY = 0
SHIP = 1
BLOCKED = 2
HIT = 3
MISS = 4

log = logging.getLogger(__name__)


class BaseGame(object):
    position_patterns = [re.compile('^([a-zа-я]+)(\d+)$', re.UNICODE),  # a1
                         re.compile('^([a-zа-я]+)\s+(\w+)$', re.UNICODE),  # a 1; a один
                         re.compile('^(\w+)\s+(\w+)$', re.UNICODE),  # a 1; a один; 7 10
                         ]

    str_letters = ['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'к']
    str_numbers = ['один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять', 'десять']

    letters_mapping = {
        'the': 'з',
        'за': 'з',
        'уже': 'ж',
        'трень': '3',
    }

    default_ships = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]

    def __init__(self):
        self.size = 0
        self.ships = None
        self.field = []
        self.enemy_field = []

        self.ships_count = 0
        self.enemy_ships_count = 0

        self.last_shot_position = None
        self.last_enemy_shot_position = None
        self.numbers = None

    def start_new_game(self, size=10, field=None, ships=None, numbers=None):
        assert(size <= 10)
        assert(len(field) == size ** 2 if field is not None else True)

        self.size = size
        self.numbers = numbers if numbers is not None else False

        if ships is None:
            self.ships = self.default_ships
        else:
            self.ships = ships

        if field is None:
            self.generate_field()
        else:
            self.field = field

        self.enemy_field = [EMPTY] * self.size ** 2

        self.ships_count = self.enemy_ships_count = len(self.ships)

        self.last_shot_position = None
        self.last_enemy_shot_position = None

    def generate_field(self):
        raise NotImplementedError()

    def print_field(self, field=None):
        if not self.size:
            log.info('Empty field')
            return

        if field is None:
            field = self.field

        mapping = [EMPTY, SHIP, BLOCKED, HIT, MISS]

        lines = ['']
        lines.append('-' * (self.size + 2))
        for y in range(self.size):
            lines.append('|%s|' % ''.join(str(mapping[x]) for x in field[y * self.size: (y + 1) * self.size]))
        lines.append('-' * (self.size + 2))
        log.info('\n'.join(lines))

    def print_enemy_field(self):
        self.print_field(self.enemy_field)

    def handle_enemy_shot(self, position):
        index = self.calc_index(position)

        if self.field[index] == SHIP:
            self.field[index] = HIT

            if self.is_dead_ship(index):
                self.ships_count -= 1
                return 'kill'
            else:
                return 'hit'
        elif self.field[index] == HIT:
            return 'kill' if self.is_dead_ship(index) else 'hit'
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
        raise NotImplementedError()

    def repeat(self):
        return self.convert_from_position(self.last_shot_position, numbers=True)

    def reset_last_shot(self):
        self.last_shot_position = None

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
            # проверяем особые случаи неправильного распознования STT
            bit = self.letters_mapping.get(bit, bit)

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

    def convert_from_position(self, position, numbers=None):
        numbers = numbers if numbers is not None else self.numbers

        if numbers:
            x = position[0]
        else:
            x = self.str_letters[position[0] - 1]

        y = position[1]

        return '%s, %s' % (x, y)


class Game(BaseGame):
    """
    Реализация игры с ипользованием DEEP MACHINE LEARNING

    Основные идеи:
    1) располагаем основную эскадру по краям площадки, в центре только 4-х палубники
    В общем и целом это дает бОльшую вероятность выиграть

    Мешаем 3 базовых возможных расположения поворотами на 90 / 180 / 270 градусов

    2) Наивная оптимизация - помечаем клетки вокруг убитых кораблей

    3) Храним список корабле противника.
    При каждом ходе помечаем места, куда могут встать оставшиеся корабли
    Чем больше кораблей могут встать в точку, тем больше ее ценность
    Выбираем для обстрела наиболее "жирные" точки

    4) При попадании в корабль меняем логику расчета матрицы с весами точек
    Теперь точка является ценной тогда и только тогда,
    когда корабль, расположенный в ней покрывает все попадания по этому кораблю

    """

    def __init__(self):
        super(Game, self).__init__()

        # Оставшиеся корабли противника
        self.enemy_ships = None

        # Текущий шаг
        self.step = 0

        # Текущий корабль противника, который будем добивать (координаты)
        self.current_enemy_ship = []

    def start_new_game(self, *args, **kwargs):
        super(Game, self).start_new_game(*args, **kwargs)

        # Ведь мы же в равных условиях? )
        self.enemy_ships = self.ships

    def generate_field(self):
        """Метод генерации поля"""

        self.field = [0] * self.size ** 2

        n = random.choice([1, 2, 3])
        # Чит-режимы, когда основные корабли помещаются по бокам
        # А однопалубники в центр
        getattr(self, 'cheat_%s' % n)()

        for i in range(random.choice([0, 1, 2, 3])):
            self.rotate_field()

        for i in range(4):
            # Однопалубники
            self.place_ship_by_indexes([random.choice([i for i, x in enumerate(self.field) if x == EMPTY])])

        for i in range(0, len(self.field)):
            if self.field[i] == BLOCKED:
                self.field[i] = EMPTY

        self.print_field()

    def place_ship_by_indexes(self, ship):
        # На вход принимаем массив клеток корабля
        # Ставим их на поле и помечаем соседние как заблокированные
        for v in ship:
            self.field[v] = SHIP
        self.mark_blocked(ship, field_attr='field')

    def cheat_1(self):
        self.place_ship_by_indexes([0, 10, 20, 30])
        self.place_ship_by_indexes([50, 60])
        self.place_ship_by_indexes([80, 90])
        self.place_ship_by_indexes([2, 12, 22])
        self.place_ship_by_indexes([42, 52, 62])
        self.place_ship_by_indexes([82, 92])

    def cheat_2(self):
        self.place_ship_by_indexes([0, 10, 20, 30])
        self.place_ship_by_indexes([50, 60, 70])
        self.place_ship_by_indexes([90, 91])
        self.place_ship_by_indexes([2, 3, 4])
        self.place_ship_by_indexes([6, 7])
        self.place_ship_by_indexes([9, 19])

    def cheat_3(self):
        self.place_ship_by_indexes([0, 10, 20, 30])
        self.place_ship_by_indexes([50, 60])
        self.place_ship_by_indexes([80, 90])
        self.place_ship_by_indexes([9, 19, 29])
        self.place_ship_by_indexes([49, 59])
        self.place_ship_by_indexes([79, 89, 99])

    def rotate_field(self, field_name='field'):
        # Поворачиваем поле
        field = getattr(self, field_name)
        for x in range(self.size // 2):
            for y in range(x, self.size - x - 1):
                i1 = self.calc_index((x + 1, y + 1))
                temp = field[i1]

                i2 = self.calc_index((y + 1, self.size - x))
                field[i1] = field[i2]

                i3 = self.calc_index((self.size - x, self.size - y))
                field[i2] = field[i3]

                i4 = self.calc_index((self.size - y, x + 1))
                field[i3] = field[i4]

                field[i4] = temp

    def do_shot(self):
        """
        Deep machine learning
        """

        # Первые 3 хода фигачим рандомом, чтобы враг расслабился
        self.step += 1
        if self.step <= 3 and not self.current_enemy_ship:
            index = random.choice([i for i, v in enumerate(self.enemy_field) if v == EMPTY])
            self.last_shot_position = self.calc_position(index)
            return self.convert_from_position(self.last_shot_position)

        # тепловая карта текущего поля врага
        value_map = [0] * self.size ** 2
        for ship in self.enemy_ships:
            for x in range(self.size):
                for y in range(self.size):
                    for direction in [1, self.size]:

                        start = self.calc_index((x + 1, y + 1))

                        if direction != 1:
                            end = self.size ** 2
                        else:
                            end = start + self.size - start % self.size

                        values = self.enemy_field[start:end:direction][:ship]
                        if len(values) < ship:
                            continue

                        if self.current_enemy_ship:
                            # Режим добивания
                            if set(self.current_enemy_ship) < set(range(start, end, direction)[:ship]):
                                for k in range(ship):
                                    if self.enemy_field[start + k * direction] == EMPTY:
                                        value_map[start + k * direction] += 1
                        else:
                            # Режим охоты
                            if not any(values):
                                for k in range(ship):
                                    value_map[start + k * direction] += 1

        for index in self.current_enemy_ship:
            # бесполезно стрелять по hit - клеткам из режима охоты, хотя они могут иметь максимальное значение
            value_map[index] = 0

        d = []
        m = 0
        # Выбираем самую сочную клетку для удара
        for i, val in enumerate(value_map):
            if val > m:
                m = val
                d = [i]
            elif val == m:
                d.append(i)

        if not d:
            # Фоллбек на всякий пожарный
            d = [i for i, v in enumerate(self.enemy_field) if v == EMPTY]

        # На всякий случай дополнительная проверка, что еще не стреляли по этому полю, там не корабль
        # и оно не заблокировано
        d = [i for i in d if self.enemy_field[i] == EMPTY]
        if d:
            index = random.choice(d)
        else:
            print('Another player is tricking me, no place to shoot')
            self.print_enemy_field()
            index = 0

        self.last_shot_position = self.calc_position(index)
        return self.convert_from_position(self.last_shot_position)

    def mark_blocked(self, ship, field_attr='enemy_field'):
        field = getattr(self, field_attr)
        # Помечаем окружающие клетки как недоступные
        sqsize = self.size ** 2

        if len(ship) > 1 \
                and ship[0] + 1 == ship[1]:
            # горизонтально
            for index in [ship[0] - 1] + \
                         ship + \
                         [ship[-1] + 1]:
                if not 0 <= index < sqsize:
                    continue

                # Для пометки нужно быть в той же строке, иначе съезжаем
                if index // self.size == ship[0] // self.size:
                    # Ставим сверху, если не верхний
                    if index - self.size >= 0:
                        field[index - self.size] = BLOCKED

                    # Ставим снизу, если не крайний нижний
                    if index + self.size < sqsize:
                        field[index + self.size] = BLOCKED

            # Ставим справа
            if ship[-1] == 0 or (ship[-1] + 1) % self.size != 0:
                field[ship[-1] + 1] = BLOCKED

            # Ставим слева
            if ship[0] % self.size != 0:
                field[ship[0] - 1] = BLOCKED

        else:

            # вериткально или одиночный + диагональ
            for index in [ship[0] - self.size] + \
                         ship + \
                         [ship[-1] + self.size]:
                if not 0 <= index < sqsize:
                    continue

                # ставим метку справа, если не крайний правый
                if index == 0 or (index + 1) % self.size != 0:
                    field[index + 1] = BLOCKED
                # Ставим метку слева, если не крайний левый
                if index % self.size != 0:
                    field[index - 1] = BLOCKED

            # Помечаем сверху
            if ship[0] - self.size >= 0:
                field[ship[0] - self.size] = BLOCKED

            # Помечаем снизу
            if ship[-1] + self.size < sqsize:
                field[ship[-1] + self.size] = BLOCKED

    def handle_enemy_reply(self, message):
        if self.last_shot_position is None:
            return

        index = self.calc_index(self.last_shot_position)

        if message in ['hit', 'kill']:
            self.current_enemy_ship.append(index)

            self.enemy_field[index] = SHIP

            if message == 'kill':
                # Больше не стреляем по клеткам, где был корабль
                self.mark_blocked(self.current_enemy_ship)
                # Больше не пытаемя поставить этот корабль на свободные места, чтобы не сбивать веса
                try:
                    self.enemy_ships.remove(len(self.current_enemy_ship))
                except ValueError:
                    print('Another alice is cheating! We have already killed all ships of this type')
                self.current_enemy_ship = []
                self.enemy_ships_count -= 1

        elif message == 'miss':
            self.enemy_field[index] = MISS

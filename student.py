import math
import heapq
import asyncio
import getpass
import json
import os
from enum import IntEnum
from typing import Optional
from consts import Tiles, Direction
import websockets
from collections import deque


DIR_KEY = ["w", "d", "s", "a"]
UNKNOWN_TILE = Tiles.SNAKE.value + 1

DIR_INCREMENTS = [
    (0, -1),  # NORTH
    (1, 0),  # EAST
    (0, 1),  # SOUTH
    (-1, 0),  # WEST
]

POSSIBLE_DIR = [
    [Direction.NORTH, Direction.WEST, Direction.EAST],  # NORTH
    [Direction.EAST, Direction.SOUTH, Direction.NORTH],  # EAST
    [Direction.SOUTH, Direction.WEST, Direction.EAST],  # SOUTH
    [Direction.WEST, Direction.SOUTH, Direction.NORTH],  # WEST
]


map_edges = set()
stones = set()
super_foods = set()


class InterestPointType(IntEnum):
    PATROL = 0
    SUPER = 1
    FOOD = 2


class GameState:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

        self.traverse = True
        self.dir = Direction.EAST
        self.current_x = 0
        self.current_y = 0
        self.range = 3
        self.score = 1
        self.step = 0

        self.map = [UNKNOWN_TILE] * self.map_cells()
        self.body_positions = set()
        self.snake_positions = set()
        self.recent_positions = deque(maxlen=30)
        self.mapped_cells = [False] * self.map_cells()

    def update_body_positions(self, body: list[tuple[int, int]]) -> None:
        self.body_positions = {tuple(segment) for segment in body}
        self.recent_positions.append((self.current_x, self.current_y))

    def map_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def map_cells(self) -> int:
        return self.width * self.height

    def _linearize(self, x: int, y: int):
        return x + y * self.width
    
    def is_valid_move(self, x: int, y: int) -> bool:
        if not self.traverse:
            if (x, y) in map_edges or (x, y) in stones:
                return False
            
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        if (x, y) in self.body_positions or (x, y) in self.snake_positions:
            return False
        for sx, sy in self.snake_positions:
            if (x, y) in self.recent_positions:
                return False
            if math.dist((x, y), (sx, sy)) < 2:
                return False

        if (x, y) in self.recent_positions:
            return False
        
        tile = self.map[self._linearize(x, y)]

        if tile == Tiles.SUPER.value and self.traverse and self.range >= 5:
            if len(super_foods) < 5:
                return False
        return True


    
    def update_map(
            self, sight: dict[str, dict[str, int]]
        ) -> list[tuple[InterestPointType, int, int]]:
            points_of_interest = []

            self.snake_positions.clear()

            for col, cols in sight.items():
                for line, tile in cols.items():
                    idx = self._linearize(int(col), int(line))

                    self.mapped_cells[idx] = True

                    #Guardar posição de passage
                    if tile == Tiles.PASSAGE.value:
                        self.map[idx] = Tiles.PASSAGE.value

                    #Guardar posição de rochas
                    if tile == Tiles.STONE.value:
                        stones.add((int(col), int(line)))
                        self.map[idx] = Tiles.STONE.value

                    #Guardar posição de outras cobras
                    if tile == Tiles.SNAKE.value and (int(col), int(line)) not in self.body_positions:
                        self.snake_positions.add((int(col), int(line)))
                        self.map[idx] = Tiles.SNAKE.value   

                    #Adiciona Food aos pontos de interesse
                    if tile == Tiles.FOOD.value:
                        points_of_interest.append((InterestPointType.FOOD, int(col), int(line)))
                        self.map[idx] = Tiles.FOOD.value

                    #Adiciona Super aos pontos de interesse só quando o traverse ta false ou a visao ta menor que 5
                    if tile == Tiles.SUPER.value:
                        super_foods.add((int(col), int(line)))
                        if self.traverse and self.range >= 5:
                            self.map[idx] = Tiles.SUPER.value 
                        else:
                            points_of_interest.append((InterestPointType.SUPER, int(col), int(line)))
                            self.map[idx] = Tiles.SUPER.value

                    #Guardar as bordas do mapa num set
                    if int(col) < 0 or int(col) >= self.width or int(line) < 0 or int(line) >= self.height:
                        if (int(col), int(line)) not in map_edges:
                            map_edges.add((int(col), int(line)))

            if all(self.mapped_cells):
                if not points_of_interest:
                    self.mapped_cells = [False] * self.map_cells()

            return points_of_interest

    def get_tile(self, x: int, y: int):
        if self.traverse:
            x = x % self.width
            y = y % self.height
        
        return self.map[self._linearize(x, y)]

    # verificar se a SUPER está rodeada de 3 obdtáculos
    def is_food_safe(self, x: int, y: int):
        count = 0
        neighbor_status = []

        neighbors = [
            ((x + 1) % self.width, y),       # Right
            ((x - 1) % self.width, y),       # Left (wraps around for 0 -> width-1)
            (x, (y + 1) % self.height),      # Down
            (x, (y - 1) % self.height),      # Up (wraps around for 0 -> height-1)
        ]


        for nx, ny in neighbors:
            if not self.traverse:
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    neighbor_status.append("OUT_OF_BOUNDS")
                    count += 1  #fora do map conta como obstaculo quando o traverse esta false
                    continue

            
            if self.get_tile(nx, ny) == Tiles.STONE.value or self.get_tile(nx, ny) == Tiles.SNAKE.value:
                neighbor_status.append("STONE/SNAKE")
                count += 1 #Stone ou corpo da cobra contam como obstaculos
            else:
                neighbor_status.append("SAFE")

        # Comida só é segura se tiver menos que 3 obstaculos
        if count < 3:
            return True
        return False




class PatrolController:
    def __init__(self, state: GameState) -> None:
        self.last_seen = [0] * state.map_cells()

    def tick(self, state: GameState, sight: dict[str, dict[str, int]]) -> None:
        for i in range(state.map_cells()):
            self.last_seen[i] += 1

        for col, cols in sight.items():
            for line, tile in cols.items():
                if tile <= Tiles.STONE.value:
                    self.last_seen[state._linearize(int(col), int(line))] = 0

    def find_next_patrol_point(self, state: GameState):
        oldest = 0
        oldest_point = (0, 0)

        for i in range(state.map_cells()):
            x = i % state.width
            y = i // state.width

            if not state.traverse and state.get_tile(x, y) != Tiles.PASSAGE:
                continue
            if (x, y) in state.recent_positions:
                continue  # evita posiçoes visitadas recentemente

            sum_seen = 0
            for x_inc in range(-state.range, state.range):
                for y_inc in range(-state.range, state.range):
                    tile_x = (x + x_inc) % state.width
                    tile_y = (y + y_inc) % state.height
                    sum_seen += self.last_seen[state._linearize(tile_x, tile_y)]

            if oldest < sum_seen:
                oldest = sum_seen
                oldest_point = (x, y)

        return oldest_point

class SearchNode:
    def __init__(self, x, y, dir, parent, heuristic):
        self.x = x
        self.y = y
        self.dir = dir
        self.parent = parent
        self.depth = parent.depth + 1 if parent is not None else 0
        self.heuristic = heuristic
        self.cost = self.depth + self.heuristic

    def __lt__(self, other):
        return self.cost < other.cost

    def __str__(self):
        return f"no({self.dir}, {self.x}, {self.y}, {self.parent})"

    def __repr__(self):
        return str(self)

    def in_parent(self, x, y):
        if self.parent is None:
            return False

        if self.parent.x == x and self.parent.y == y:
            return True

        return self.parent.in_parent(x, y)




class PathFind:
    def path_to_target(self, state: GameState, goal_x: int, goal_y: int) -> Optional[list[SearchNode]]:
        open_set = []
        start_node = SearchNode(state.current_x, state.current_y, state.dir, None, self._heuristic(state, state.current_x, state.current_y, goal_x, goal_y))
        heapq.heappush(open_set, (start_node.cost, start_node))

        tiles_visited = {(start_node.x, start_node.y): start_node.cost}

        while open_set:
            _, current_node = heapq.heappop(open_set)

            if current_node.x == goal_x and current_node.y == goal_y:
                return self.get_path(current_node)

            for new_dir in POSSIBLE_DIR[current_node.dir]:
                (x_inc, y_inc) = DIR_INCREMENTS[new_dir]
                new_x = current_node.x + x_inc
                new_y = current_node.y + y_inc



                if state.traverse:
                    new_x %= state.width
                    new_y %= state.height
                else:
                    if new_x < 0 or new_x >= state.width or new_y < 0 or new_y >= state.height:
                        continue
                if not state.is_valid_move(new_x, new_y):
                    continue

                if (new_x, new_y) in super_foods:
                    super_foods.remove((new_x, new_y))

                heuristic = self._heuristic(state, new_x, new_y, goal_x, goal_y)
                new_node = SearchNode(new_x, new_y, new_dir, current_node, heuristic)

                # ver se ha um melhor caminho para o no
                if (new_x, new_y) not in tiles_visited or new_node.cost < tiles_visited[(new_x, new_y)]:
                    tiles_visited[(new_x, new_y)] = new_node.cost
                    heapq.heappush(open_set, (new_node.cost, new_node))

        return None 

    def _heuristic(self, state: GameState, x: int, y: int, goal_x: int, goal_y: int) -> float:
        distance = math.dist((x, y), (goal_x, goal_y))
        unexplored_bonus = -20 if state.mapped_cells[state._linearize(goal_x, goal_y)] else 0
        return distance + unexplored_bonus

    def get_path(self, node: SearchNode) -> list[SearchNode]:
        path = []
        while node:
            path.append(node)
            node = node.parent
        path.reverse()  # set path da posição atual ao objetivo
        return path

class Agent:
    def __init__(self, initial_state) -> None:
        self.game_state = GameState(initial_state["size"][0], initial_state["size"][1])
        self.path_find = PathFind()
        self.patrol_controller = PatrolController(self.game_state)

        self.interest_points: set[tuple[InterestPointType, int, int]] = set()
        self.current_target: Optional[tuple[InterestPointType, int, int]] = None
        self.current_path: Optional[list[SearchNode]] = None 

    def is_patrol_point(self) -> bool:
        return (
            self.current_target is not None
            and self.current_target[0] == InterestPointType.PATROL
        )

    def invalidate_patrol_point(self):
        self.current_target = None
        self.current_path = None

    def tick(self, state) -> Optional[str]:
        if "body" not in state:
            return None

        self.game_state.score = state.get("score", self.game_state.score)   
        self.game_state.step = state.get("step", self.game_state.step)
        self.game_state.current_x = state["body"][0][0]
        self.game_state.current_y = state["body"][0][1]
        self.game_state.update_body_positions(state["body"])
        new_interest_points = self.game_state.update_map(state.get("sight", {}))

        if new_interest_points:
        
            self.invalidate_patrol_point()
            self.interest_points.update(new_interest_points)

        if self.game_state.traverse != state.get("traverse", self.game_state.traverse):
            self.game_state.traverse = state["traverse"]
            self.invalidate_patrol_point()

            if self.interest_points:
                valid_targets = [
                    ip for ip in self.interest_points
                    if self.game_state.is_valid_move(ip[1], ip[2])
                ]
                if valid_targets:
                    self.current_target = min(
                        valid_targets,
                        key=lambda ip: (ip[0].value, math.dist(
                            (self.game_state.current_x, self.game_state.current_y),
                            (ip[1], ip[2])
                        ))
                    )
                    self.interest_points.remove(self.current_target)
                    
            if self.current_target:
                if not self.game_state.is_valid_move(self.current_target[1], self.current_target[2]):
                    self.current_target = None

        if self.game_state.range != state.get("range", self.game_state.range):
            self.game_state.range = state["range"]
            self.invalidate_patrol_point()

        # Atualizal patrol controller
        self.patrol_controller.tick(self.game_state, state.get("sight", {}))

        # Apagar tile objetivo se a cobra for a esse tile
        if self.current_target and (self.game_state.current_x, self.game_state.current_y) == (self.current_target[1], self.current_target[2]):
            self.current_target = None

        # selecionar target se nao houver nenhum
        if not self.current_target:
            if self.interest_points:
                # Priorizar targets dependendo do tipo
                self.current_target = min(
                    self.interest_points,
                    key=lambda ip: (ip[0].value, math.dist((self.game_state.current_x, self.game_state.current_y), (ip[1], ip[2])))
                )
                self.interest_points.remove(self.current_target)
            else:
                # definir target patrol
                tgt_x, tgt_y = self.patrol_controller.find_next_patrol_point(self.game_state)
                self.current_target = (InterestPointType.PATROL, tgt_x, tgt_y)

        # encontrar caminho para o target
        path = self.path_find.path_to_target(self.game_state, self.current_target[1], self.current_target[2])

        if path is None:
            # Definir proximo passo caso nao haja nenhum caminho 
            for i, (x_inc, y_inc) in enumerate(DIR_INCREMENTS):
                next_x = self.game_state.current_x + x_inc
                next_y = self.game_state.current_y + y_inc

                if self.game_state.is_valid_move(next_x, next_y):
                    key = DIR_KEY[i]
                    return key

            return None

        # fazer o passo seguinte no caminho encontrado
        if len(path) > 1:
            self.game_state.dir = path[1].dir
            key = DIR_KEY[path[1].dir]
            return key

        return None

async def agent_loop(server_address="localhost:8000", agent_name="student"):
    async with websockets.connect(f"ws://{server_address}/player") as websocket:
        # Receive information about static game properties
        await websocket.send(json.dumps({"cmd": "join", "name": agent_name}))

        initial_state = json.loads(await websocket.recv())
        print(initial_state)

        agent = Agent(initial_state)

        while True:
            try:
                state = json.loads(await websocket.recv())

                key = agent.tick(state)

                if key is not None:
                    await websocket.send(json.dumps({"cmd": "key", "key": key}))
            except websockets.exceptions.ConnectionClosedOK:
                print("Server has cleanly disconnected us")
                return


# DO NOT CHANGE THE LINES BELOW
# You can change the default values using the command line, example:
# $ NAME='arrumador' python3 client.py
loop = asyncio.get_event_loop()
SERVER = os.environ.get("SERVER", "localhost")
PORT = os.environ.get("PORT", "8000")
NAME = os.environ.get("NAME", getpass.getuser())
loop.run_until_complete(agent_loop(f"{SERVER}:{PORT}", NAME))
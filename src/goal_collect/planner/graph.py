import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.prepared import prep
from ..config import config


@dataclass(frozen=True)
class Node:
    x: float
    y: float
    id: int = field(compare=False)


class Graph:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[int, Dict[int, float]] = {}  # u_id -> {v_id: weight}
        self.poi_nodes: Dict[str, List[int]] = {}  # poi_id -> list of node_ids inside the POI
        self._next_id = 0
        self.spatial_bins: Dict[Tuple[int, int], List[int]] = {}
        self.spatial_bin_size: float = 2.0

    def build_spatial_index(self, bin_size: float = 2.0):
        """Builds a spatial hash for O(1) avg neighbor lookups."""
        self.spatial_bin_size = bin_size
        self.spatial_bins = {}
        for nid, n in self.nodes.items():
            bx = math.floor(n.x / bin_size)
            by = math.floor(n.y / bin_size)
            if (bx, by) not in self.spatial_bins:
                self.spatial_bins[(bx, by)] = []
            self.spatial_bins[(bx, by)].append(nid)

    def get_nodes_in_radius(self, x: float, y: float, r: float) -> List[int]:
        """Finds all nodes within radius r of (x,y)."""
        if not self.spatial_bins:
            self.build_spatial_index(max(r, 2.0))

        candidates = []
        # Check tolerance to include bins involved
        # Use simple bounding box around (x,y) with radius r
        min_bx = math.floor((x - r) / self.spatial_bin_size)
        max_bx = math.floor((x + r) / self.spatial_bin_size)
        min_by = math.floor((y - r) / self.spatial_bin_size)
        max_by = math.floor((y + r) / self.spatial_bin_size)

        r_sq = r * r

        for bx in range(min_bx, max_bx + 1):
            for by in range(min_by, max_by + 1):
                key = (bx, by)
                if key in self.spatial_bins:
                    for nid in self.spatial_bins[key]:
                        n = self.nodes[nid]
                        if (n.x - x) ** 2 + (n.y - y) ** 2 <= r_sq:
                            candidates.append(nid)
        return candidates

    def add_node(self, x: float, y: float) -> int:
        node_id = self._next_id
        self._next_id += 1
        self.nodes[node_id] = Node(x, y, node_id)
        self.edges[node_id] = {}
        return node_id

    def add_edge(self, u_id: int, v_id: int, weight: float):
        self.edges[u_id][v_id] = weight
        self.edges[v_id][u_id] = weight

    def remove_node(self, node_id: int):
        if node_id not in self.nodes:
            return

        # Remove edges from neighbors
        for v_id in self.edges[node_id]:
            if node_id in self.edges[v_id]:
                del self.edges[v_id][node_id]

        # Remove node's edge list
        del self.edges[node_id]
        # Remove node
        del self.nodes[node_id]

    def get_neighbors(self, u_id: int) -> Dict[int, float]:
        return self.edges.get(u_id, {})

    def get_node(self, node_id: int) -> Node:
        return self.nodes[node_id]

    def get_closest_node(self, x: float, y: float) -> int:
        """Finds the node closest to the given coordinates."""
        if not self.nodes:
            raise ValueError("Graph has no nodes")

        closest_id = -1
        min_dist = float("inf")

        for node in self.nodes.values():
            dist = np.hypot(node.x - x, node.y - y)
            if dist < min_dist:
                min_dist = dist
                closest_id = node.id

        return closest_id

    def get_closest_node_in_poi(self, x: float, y: float, poi_id: str) -> Node:
        """
        Returns the node within the poi that is closest to the x,y point.
        """
        if poi_id not in self.poi_nodes:
            raise ValueError(f"POI {poi_id} not found in graph")

        candidate_ids = self.poi_nodes[poi_id]
        if not candidate_ids:
            raise ValueError(f"POI {poi_id} has no associated nodes")

        closest_id = -1
        min_dist = float("inf")

        for node_id in candidate_ids:
            node = self.nodes[node_id]
            dist = np.hypot(node.x - x, node.y - y)
            if dist < min_dist:
                min_dist = dist
                closest_id = node_id

        return self.nodes[closest_id]


def get_poly(obs) -> Polygon:
    if isinstance(obs, Polygon):
        return obs
    if hasattr(obs, "polygon"):
        return obs.polygon
    if hasattr(obs, "shape"):
        return obs.shape
    raise ValueError(f"Unknown obstacle type: {type(obs)}")


class EnvironmentMap:
    def __init__(
        self,
        static_obstacles: Sequence[object],
        goals: Optional[Sequence[object]] = None,
        collection_points: Optional[Sequence[object]] = None,
    ):
        self.goals = goals
        self.collection_points = collection_points
        self.robot_radius = config.agent.robot_radius + config.graph.robot_radius_buffer

        # Determine obstacles vs boundary
        # The boundary is defined as a LinearRing, while obstacles are Polygons.
        # We need to separate them.
        self.obstacles = []
        self.boundary_poly = None

        for obs in static_obstacles:
            # Get the underlying shape
            shape = getattr(obs, "shape", obs)
            poly = get_poly(obs)

            # Check if it was originally a LinearRing (boundary)
            # get_poly converts to Polygon/shape, so we check original object or shape type
            if isinstance(shape, LineString) or (hasattr(shape, "geom_type") and shape.geom_type == "LinearRing"):
                # It's a boundary. Convert to Polygon for .contains() check.
                # Note: LinearRing in shapely is subclass of LineString but behaves like closed ring.
                b_poly = Polygon(poly)
                if self.robot_radius > 0:
                    b_poly = b_poly.buffer(-self.robot_radius)
                self.boundary_poly = b_poly
            else:
                if self.robot_radius > 0:
                    poly = poly.buffer(self.robot_radius)
                self.obstacles.append(poly)

        self.prepared_obstacles = [prep(p) for p in self.obstacles]

        if self.boundary_poly:
            self.prepared_boundary = prep(self.boundary_poly)
        else:
            raise ValueError("No boundary found in environment map")

    def is_free(self, x: float, y: float) -> bool:
        p = Point(x, y)

        # Check boundary first
        if self.boundary_poly:
            if not self.boundary_poly.contains(p):
                return False

        # Check obstacles
        for prepared_obs in self.prepared_obstacles:
            if prepared_obs.intersects(p):
                return False
        return True

    def is_segment_free(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        line = LineString([(x1, y1), (x2, y2)])
        for obs in self.prepared_obstacles:
            if obs.intersects(line):
                return False
        return True


def generate_occupancy_grid(
    env_map: EnvironmentMap, 
    resolution: float,
    start_positions: Optional[List[Tuple[float, float]]] = None
) -> Graph:
    """
    Generates a graph based on an occupancy grid.
    Nodes are centers of free cells.
    Edges connect adjacent free cells (8-connectivity).
    Agent start positions are added as exact nodes connected to nearby grid cells.
    """
    graph = Graph()

    min_x, min_y, max_x, max_y = env_map.boundary_poly.bounds

    xs = np.arange(min_x, max_x, resolution)
    ys = np.arange(min_y, max_y, resolution)

    # Store node_ids in a grid to easily add edges
    grid_shape = (len(ys), len(xs))
    node_grid = np.full(grid_shape, -1, dtype=int)

    # 1. Create Nodes
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            # Check center of the cell
            cx, cy = x + resolution / 2, y + resolution / 2
            if env_map.is_free(cx, cy):
                node_id = graph.add_node(cx, cy)
                node_grid[j, i] = node_id

    # 2. Create Edges
    # 8-connectivity
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]  # Cardinal  # Diagonal
    diag_dist = np.sqrt(2) * resolution
    card_dist = resolution

    # Iterate over all valid nodes
    for j in range(grid_shape[0]):
        for i in range(grid_shape[1]):
            u_id = node_grid[j, i]
            if u_id == -1:
                continue

            for dj, di in moves:
                nj, ni = j + dj, i + di
                if 0 <= nj < grid_shape[0] and 0 <= ni < grid_shape[1]:
                    v_id = node_grid[nj, ni]
                    if v_id != -1:
                        # Determine weight
                        weight = diag_dist if (dj != 0 and di != 0) else card_dist
                        graph.add_edge(u_id, v_id, weight)

    # 2.5. Add Agent Start Positions as exact nodes (if not already in grid)
    if start_positions:
        for sx, sy in start_positions:
            if env_map.is_free(sx, sy):
                # Check if a node already exists very close to this position
                existing_node_id = None
                for node_id, node in graph.nodes.items():
                    if abs(node.x - sx) < 1e-4 and abs(node.y - sy) < 1e-4:
                        existing_node_id = node_id
                        break
                
                if existing_node_id is not None:
                    # Node already exists at this position, no need to add
                    continue
                
                # Add the exact start position as a new node
                start_node_id = graph.add_node(sx, sy)
                
                # Connect to nearby grid nodes (within ~2 grid cells)
                connection_radius = resolution * 2.5
                for node_id, node in graph.nodes.items():
                    if node_id == start_node_id:
                        continue
                    dist = np.hypot(node.x - sx, node.y - sy)
                    if dist <= connection_radius:
                        # Check if path is free (should be in same grid area)
                        if env_map.is_segment_free(sx, sy, node.x, node.y):
                            graph.add_edge(start_node_id, node_id, dist)

    # 3. Associate Nodes with POIs (Goals/Collection Points)
    pois = []
    if env_map.goals:
        pois.extend(env_map.goals)
    if env_map.collection_points:
        pois.extend(env_map.collection_points)

    for poi in pois:
        poi_id = getattr(poi, "id", getattr(poi, "goal_id", getattr(poi, "point_id", None)))
        if poi_id is None:
            continue
        poi_poly = get_poly(poi)
        # Apply small negative tolerance to exclude boundary nodes
        search_poly = poi_poly.buffer(-config.graph.poi_buffer)

        # Find all nodes inside this POI
        inside_nodes = []
        for node in graph.nodes.values():
            if search_poly.contains(Point(node.x, node.y)):
                inside_nodes.append(node.id)

        # Always add the center node (centroid) for precise targeting
        centroid = poi_poly.centroid

        # Strategy: Replace the closest existing graph node with this centroid node
        # This keeps the graph structure but moves a node to the exact center.

        # 1. Find the closest node
        closest_id = graph.get_closest_node(centroid.x, centroid.y)

        if closest_id != -1:
            closest_node = graph.get_node(closest_id)

            # Check if this node is already a centroid (from another POI)?
            # If so, avoiding touching it or just letting it be.
            # Assuming POIs don't overlap significantly for their centers.

            # 2. Add the centroid node
            c_id = graph.add_node(centroid.x, centroid.y)

            # 3. Transfer edges
            # Connect c_id to all neighbors of closest_id
            neighbors = graph.get_neighbors(closest_id)
            for neighbor_id, _ in neighbors.items():
                # Re-calculate weight based on new distance
                neighbor_node = graph.get_node(neighbor_id)
                new_weight = np.hypot(neighbor_node.x - centroid.x, neighbor_node.y - centroid.y)
                graph.add_edge(c_id, neighbor_id, new_weight)

            # 4. Remove the old closest node
            graph.remove_node(closest_id)

            # 5. Update POI lists?
            # We are building `inside_nodes` right now.
            # The `closest_id` might have been in `inside_nodes` list (if we calculated it before).
            # But we calculate `inside_nodes` fresh here?
            # Wait, the code above calculates `inside_nodes`.
            # We should probably filter `inside_nodes` to remove `closest_id` if it's there.

            # Refine `inside_nodes` list
            # It currently contains `closest_id` if it was inside.
            inside_nodes = [nid for nid in inside_nodes if nid != closest_id]
            inside_nodes.insert(0, c_id)

        else:
            # Fallback (empty graph?): Just add c_id
            c_id = graph.add_node(centroid.x, centroid.y)
            inside_nodes.insert(0, c_id)

        graph.poi_nodes[poi_id] = inside_nodes

        graph.poi_nodes[poi_id] = inside_nodes

    graph.build_spatial_index()
    return graph


def generate_prm(env_map: EnvironmentMap, num_samples: int, connection_radius: float) -> Graph:
    """
    Generates a PRM graph.
    """
    graph = Graph()

    # 1. Sample Nodes
    count = 0
    rng = np.random.default_rng(config.graph.prm_seed)  # Fixed seed for reproducibility

    min_x, min_y, max_x, max_y = env_map.boundary_poly.bounds
    while count < num_samples:
        rx = rng.uniform(min_x, max_x)
        ry = rng.uniform(min_y, max_y)

        if env_map.is_free(rx, ry):
            graph.add_node(rx, ry)
            count += 1

    # 2. Connect Nodes
    # Simple O(N^2) approach for now, or use KDTree if optimization needed.
    # Given simulation scale, N might be small (e.g. 100-500). O(N^2) is fine.

    node_ids = list(graph.nodes.keys())
    for i in range(len(node_ids)):
        u_id = node_ids[i]
        u = graph.get_node(u_id)

        for j in range(i + 1, len(node_ids)):
            v_id = node_ids[j]
            v = graph.get_node(v_id)

            dist = np.hypot(u.x - v.x, u.y - v.y)
            if dist <= connection_radius:
                if env_map.is_segment_free(u.x, u.y, v.x, v.y):
                    graph.add_edge(u_id, v_id, dist)

    # 3. Associate Nodes with POIs (Goals/Collection Points)
    pois = []
    if env_map.goals:
        pois.extend(env_map.goals)
    if env_map.collection_points:
        pois.extend(env_map.collection_points)

    for poi in pois:
        poi_id = getattr(poi, "id", getattr(poi, "goal_id", getattr(poi, "point_id", None)))
        if poi_id is None:
            continue
        poi_poly = get_poly(poi)
        # Apply small negative tolerance to exclude boundary nodes
        search_poly = poi_poly.buffer(-config.graph.poi_buffer)

        # Find all nodes inside this POI
        inside_nodes = []
        for node in graph.nodes.values():
            if search_poly.contains(Point(node.x, node.y)):
                inside_nodes.append(node.id)

        # Fallback: If no nodes are inside, add the centroid
        if not inside_nodes:
            centroid = poi_poly.centroid
            if env_map.is_free(centroid.x, centroid.y):
                c_id = graph.add_node(centroid.x, centroid.y)
                inside_nodes.append(c_id)

                # Connect fallback node to graph (closest visible node)
                min_d = float("inf")
                best_n = -1
                for node in graph.nodes.values():
                    if node.id == c_id:
                        continue
                    d = np.hypot(node.x - centroid.x, node.y - centroid.y)
                    # Try to find closest visible node, preferably within radius, but check all if needed
                    if d < min_d:
                        if env_map.is_segment_free(centroid.x, centroid.y, node.x, node.y):
                            min_d = d
                            best_n = node.id

                if best_n != -1:
                    graph.add_edge(c_id, best_n, min_d)

        graph.poi_nodes[poi_id] = inside_nodes

    graph.build_spatial_index()
    return graph


def generate_inflated_visibility_graph(
    env_map: EnvironmentMap, start_positions: Optional[List[Tuple[float, float]]] = None
) -> Graph:
    """
    Generates a visibility graph using inflated obstacles, POI centers, and agent start positions.
    """
    graph = Graph()

    # 1. Collect candidate nodes
    # We will use vertices of obstacles inflated by a small epsilon
    # env_map.obstacles are already buffered by robot_radius.

    epsilon = config.graph.visibility_graph_epsilon
    all_node_ids = []

    # Process obstacles to get vertices
    for obs in env_map.obstacles:
        # Buffer slightly to ensure nodes are outside the forbidden region
        # Use join_style=2 (mitre) for potentially sharper corners, though input might be round.
        # Then simplify to reduce vertex count (approximating arcs).
        # join_style=2 is 'mitre', avoiding round corners if possible/applicable
        inflated = obs.buffer(epsilon, join_style=2).simplify(0.05, preserve_topology=False)

        if isinstance(inflated, Polygon):
            polys = [inflated]
        else:
            # MultiPolygon
            polys = inflated.geoms

        for poly in polys:
            # Exterior
            vals = list(poly.exterior.coords)
            # Remove duplicate end point
            if len(vals) > 1 and vals[0] == vals[-1]:
                vals.pop()

            for x, y in vals:
                # Add node if it's within bounds (optional but safe)
                if env_map.boundary_poly.contains(Point(x, y)):
                    nid = graph.add_node(x, y)
                    all_node_ids.append(nid)

    # 2. Add POI centers (Goals and Collection Points)
    pois = []
    if env_map.goals:
        pois.extend(env_map.goals)
    if env_map.collection_points:
        pois.extend(env_map.collection_points)

    for poi in pois:
        poi_poly = get_poly(poi)
        centroid = poi_poly.centroid

        # POI centroids are mission targets - add them unconditionally
        # They must be reachable even if inside a buffered obstacle zone
        nid = graph.add_node(centroid.x, centroid.y)
        all_node_ids.append(nid)

    # Add agent start positions
    if start_positions:
        for x, y in start_positions:
            if env_map.is_free(x, y):
                nid = graph.add_node(x, y)
                all_node_ids.append(nid)

    # 3. Connect Nodes (Visibility Check)
    # Check all pairs O(N^2)
    node_ids = list(graph.nodes.keys())
    for i in range(len(node_ids)):
        u_id = node_ids[i]
        u = graph.get_node(u_id)

        for j in range(i + 1, len(node_ids)):
            v_id = node_ids[j]
            v = graph.get_node(v_id)

            # Distance
            dist = np.hypot(u.x - v.x, u.y - v.y)

            # Check visibility
            if env_map.is_segment_free(u.x, u.y, v.x, v.y):
                graph.add_edge(u_id, v_id, dist)

    # 4. Optional: Reduce Visibility Graph
    if config.graph.reduced_visibility_graph:
        node_ids = list(graph.nodes.keys())
        edges_to_remove = []

        # We look for triangles (u, v, w) where edge (u, w) is redundant because path u->v->w is almost same length
        # Iterate over all nodes 'v' which act as intermediate
        for v_id in graph.nodes:
            neighbors = graph.get_neighbors(v_id)
            if len(neighbors) < 2:
                continue

            # Check pairs of neighbors (u, w)
            neighbor_ids = list(neighbors.keys())
            for idx1 in range(len(neighbor_ids)):
                u_id = neighbor_ids[idx1]
                for idx2 in range(idx1 + 1, len(neighbor_ids)):
                    w_id = neighbor_ids[idx2]

                    # Check if edge (u, w) exists
                    if w_id in graph.get_neighbors(u_id):
                        # Get distances
                        d_uv = graph.edges[u_id][v_id]
                        d_vw = graph.edges[v_id][w_id]
                        d_uw = graph.edges[u_id][w_id]

                        # Check triangle inequality tightness
                        # If d_uw ~ d_uv + d_vw, then u->w is redundant
                        if d_uw > (d_uv + d_vw) * 0.999:  # Tolerance
                            edges_to_remove.append((u_id, w_id))

        # Remove edges
        for u_id, w_id in edges_to_remove:
            # Check if still exists (might have been removed already in reverse order or duplicate)
            if w_id in graph.edges[u_id]:
                # We need a proper remove_edge function or manual
                if u_id in graph.edges and w_id in graph.edges[u_id]:
                    del graph.edges[u_id][w_id]
                if w_id in graph.edges and u_id in graph.edges[w_id]:
                    del graph.edges[w_id][u_id]

    # 5. Cleanup: Remove nodes with no edges
    nodes_to_remove = []
    for node_id in graph.nodes:
        if not graph.get_neighbors(node_id):
            nodes_to_remove.append(node_id)

    for node_id in nodes_to_remove:
        # remove_node handles edge removal too, but they have no edges so fine.
        graph.remove_node(node_id)

    # 6. Associate Nodes with POIs
    for poi in pois:
        poi_id = getattr(poi, "id", getattr(poi, "goal_id", getattr(poi, "point_id", None)))
        if poi_id is None:
            continue

        poi_poly = get_poly(poi)
        search_poly = poi_poly.buffer(-config.graph.poi_buffer)
        centroid = poi_poly.centroid

        inside_nodes = []
        centroid_node_id = -1

        for node in graph.nodes.values():
            if search_poly.contains(Point(node.x, node.y)):
                inside_nodes.append(node.id)
                # Check if this node is the centroid (with small tolerance)
                if abs(node.x - centroid.x) < 1e-6 and abs(node.y - centroid.y) < 1e-6:
                    centroid_node_id = node.id

        # Ensure centroid is first
        if centroid_node_id != -1 and inside_nodes and inside_nodes[0] != centroid_node_id:
            inside_nodes.remove(centroid_node_id)
            inside_nodes.insert(0, centroid_node_id)

        graph.poi_nodes[poi_id] = inside_nodes
        if not inside_nodes:
            print(f"[WARNING] Visibility graph: POI {poi_id} has no associated nodes!")

    graph.build_spatial_index()
    return graph

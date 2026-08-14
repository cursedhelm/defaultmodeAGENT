"""Visualization page for the Agent Manager TUI."""

from typing import Optional, List, Tuple
from collections import defaultdict
import numpy as np

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.events import Click
from textual.widgets import Button, Select, Static, Checkbox, Input
from textual.worker import get_current_worker

from rich.text import Text as RichText

from tui.shared import (
    STATE, VizNode, get_bot_caches,
    reduce_dimensions_sparse,
    save_viz_cache, load_viz_cache,
    _draw_line,
)
from tui.viz_graph import VizGraph, live_bot_names, resolve_graph, save_graph_cache


def _select_is_blank(value) -> bool:
    """Handle both old and current Textual empty-select sentinels."""
    return (
        value is None
        or value == ""
        or value is Select.BLANK
        or value is getattr(Select, "NULL", None)
    )


class VizPage(Vertical):
    """Memory latent space visualization page with zoom and navigation."""

    BINDINGS = [
        Binding("w", "move_up", "Up", show=False),
        Binding("s", "move_down", "Down", show=False),
        Binding("a", "move_left", "Left", show=False),
        Binding("d", "move_right", "Right", show=False),
        Binding("up", "pan_up", "Pan Up", show=False),
        Binding("down", "pan_down", "Pan Down", show=False),
        Binding("left", "pan_left", "Pan Left", show=False),
        Binding("right", "pan_right", "Pan Right", show=False),
        Binding("enter", "select_node", "Select", show=False),
        Binding("+", "zoom_in", "Zoom+", show=False),
        Binding("=", "zoom_in", "Zoom+", show=False),
        Binding("-", "zoom_out", "Zoom-", show=False),
        Binding("f", "focus_selected", "Focus", show=False),
        Binding("left_square_bracket", "previous_in_cell", "Cell Previous", show=False),
        Binding("right_square_bracket", "next_in_cell", "Cell Next", show=False),
        Binding("n", "next_connection", "Next Connection", show=False),
        Binding("p", "previous_connection", "Previous Connection", show=False),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph: Optional[VizGraph] = None
        self.loaded_bot: Optional[str] = None
        self.nodes: List[VizNode] = []
        self.method: str = "pca"
        self.selected_idx: int = -1
        self.connections: List[int] = []
        self.zoom: float = 1.0
        self.extended_neighbors: bool = False
        self.show_context: bool = False
        self.show_themes: bool = True
        self.scope_user_id: Optional[str] = None
        self._projection_cached: bool = False
        self.view_cx: float = 0.0
        self.view_cy: float = 0.0
        self._grid_w: int = 80
        self._grid_h: int = 40
        # cached bounds — set once in _finish_viz, read on every render/pan
        self._data_min_x: float = 0.0
        self._data_max_x: float = 1.0
        self._data_min_y: float = 0.0
        self._data_max_y: float = 1.0
        self._dr_x: float = 1.0
        self._dr_y: float = 1.0
        # O(1) lookup structures rebuilt after each load
        self._mid_to_node_idx: dict = {}     # mid → index in self.nodes
        self._conn_cache: dict = {}          # (mid, top_k, scope) → neighbors
        self._visible_mids: set = set()      # mids currently in view (respects user scope)
        self._scope_node_indices: List[int] = []
        self._render_node_indices: List[int] = []
        self._cell_nodes: dict = {}
        self._themed_mids: set[int] = set()
        self._search_results: List[int] = []
        self._search_pos: int = -1
        # flat numpy coord arrays for vectorized grid projection
        self._node_xs: "np.ndarray" = np.empty(0, dtype=np.float32)
        self._node_ys: "np.ndarray" = np.empty(0, dtype=np.float32)

    def compose(self) -> ComposeResult:
        yield Vertical(
            Horizontal(
                Select([], id="viz-bot-select", prompt="select bot"),
                Select([], id="viz-user-select", prompt="all users"),
                Select(
                    [("LSA", "lsa"), ("PCA", "pca"), ("UMAP", "umap")],
                    id="viz-method", value="pca",
                ),
                id="viz-scope-controls",
            ),
            Horizontal(
                Checkbox("Context", id="viz-context"),
                Checkbox("Extended", id="viz-extended"),
                Checkbox("Themes", value=True, id="viz-themes"),
                Button("Refresh State", id="viz-load-btn"),
                Button("Rebuild Map", id="viz-refresh-btn"),
                id="viz-action-controls",
            ),
            Horizontal(
                Input(
                    placeholder="search graph text, terms, or #memory-id",
                    id="viz-search-input",
                ),
                Button("Find", id="viz-search-btn"),
                Button("Next", id="viz-search-next-btn"),
                id="viz-search-controls",
            ),
            id="viz-controls",
        )
        yield Static("", id="viz-status")
        yield Horizontal(
            Vertical(
                Static("[dim]Load memories to visualize[/dim]", id="viz-content"),
                id="viz-canvas",
            ),
            Vertical(
                Static("[bold]Memory Details[/bold]", id="viz-detail-header"),
                ScrollableContainer(
                    Static("", id="viz-detail-content"),
                    id="viz-detail-scroll",
                ),
                Static("[bold]Connections[/bold]", id="viz-conn-header"),
                ScrollableContainer(
                    Static("", id="viz-connections"),
                    id="viz-conn-scroll",
                ),
                id="viz-detail-panel",
            ),
            id="viz-main",
        )

    def on_mount(self):
        self._refresh_bot_list()

    def _refresh_bot_list(self):
        bot_names = sorted(
            set(get_bot_caches()) | set(STATE.live_bot_names()) | set(live_bot_names())
        )
        bots = [(b, b) for b in bot_names]
        sel = self.query_one("#viz-bot-select", Select)
        sel.set_options(bots)
        if bots and _select_is_blank(sel.value):
            preferred = STATE.selected_bot if STATE.selected_bot in bot_names else bots[0][1]
            sel.value = preferred

    def _refresh_user_list(self):
        users = self.graph.users if self.graph else []
        user_sel = self.query_one("#viz-user-select", Select)
        user_sel.set_options([("all users", "")] + [(u, u) for u in users])
        value = self.scope_user_id if self.scope_user_id in users else ""
        self.scope_user_id = value or None
        user_sel.value = value

    @on(Select.Changed, "#viz-method")
    def on_method_change(self, event: Select.Changed):
        if not _select_is_blank(event.value):
            self.method = event.value
            if self.graph:
                self._generate_viz()

    @on(Select.Changed, "#viz-bot-select")
    def on_bot_changed(self, event: Select.Changed):
        if _select_is_blank(event.value):
            return
        self._load_bot(event.value)

    @on(Select.Changed, "#viz-user-select")
    def on_user_changed(self, event: Select.Changed):
        if self.graph:
            self.scope_user_id = (
                None if _select_is_blank(event.value) else str(event.value)
            )
            self._apply_user_scope()

    @on(Checkbox.Changed, "#viz-context")
    def on_context_change(self, event: Checkbox.Changed):
        self.show_context = event.value
        if self.nodes:
            self._apply_user_scope(preserve_selection=True)

    @on(Checkbox.Changed, "#viz-themes")
    def on_themes_change(self, event: Checkbox.Changed):
        self.show_themes = event.value
        if self.nodes:
            self._apply_theme_scope()
            self._render_canvas()

    @on(Checkbox.Changed, "#viz-extended")
    def on_extended_change(self, event: Checkbox.Changed):
        self.extended_neighbors = event.value
        if self.nodes and 0 <= self.selected_idx < len(self.nodes):
            self._show_node_details(self.nodes[self.selected_idx])

    @on(Button.Pressed, "#viz-load-btn")
    def on_load(self):
        """Refresh from live RAM when available, otherwise the disk checkpoint."""
        sel = self.query_one("#viz-bot-select", Select)
        if _select_is_blank(sel.value):
            return
        self._load_bot(str(sel.value), force_source_reload=True)

    @on(Button.Pressed, "#viz-refresh-btn")
    def on_refresh(self):
        if self.graph:
            self._generate_viz(force_rebuild=True)

    @on(Button.Pressed, "#viz-search-btn")
    @on(Input.Submitted, "#viz-search-input")
    def on_search(self) -> None:
        graph = self.graph
        query = self.query_one("#viz-search-input", Input).value.strip()
        if graph is None or not query:
            return
        self._set_status(f"[yellow]searching graph for {query!r}…[/yellow]")
        self._search_graph_worker(
            graph, query, self.scope_user_id, graph.source_key
        )

    @on(Button.Pressed, "#viz-search-next-btn")
    def on_search_next(self) -> None:
        self._advance_search_result()

    @work(thread=True, exclusive=True, name="search_graph")
    def _search_graph_worker(
        self,
        graph: VizGraph,
        query: str,
        user_id: Optional[str],
        source_key: str,
    ) -> None:
        hits = graph.search(query, user_id=user_id, limit=500)
        self.app.call_from_thread(
            self._on_search_ready, graph, query, source_key,
            [hit.mid for hit in hits],
        )

    def _on_search_ready(
        self,
        graph: VizGraph,
        query: str,
        source_key: str,
        mids: List[int],
    ) -> None:
        if self.graph is not graph or graph.source_key != source_key:
            return
        self._search_results = mids
        self._search_pos = -1
        if not mids:
            self._update_status(extra=f"search={query!r}: no matches")
            return
        self._advance_search_result()
        self._update_status(extra=f"search={query!r}: {len(mids):,} matches")

    def _advance_search_result(self) -> None:
        if not self._search_results:
            return
        self._search_pos = (self._search_pos + 1) % len(self._search_results)
        self.select_mid(self._search_results[self._search_pos], focus=True)

    def _load_bot(self, bot_name: str, force_source_reload: bool = False):
        """Resolve a read-only graph without blocking the Textual event loop."""
        preserve_current = self.loaded_bot == bot_name and self.graph is not None
        self.loaded_bot = bot_name
        if not preserve_current:
            self.scope_user_id = None
            self.graph = None
            self.nodes = []
            self.selected_idx = -1
            self.connections = []
            self._scope_node_indices = []
            self._render_node_indices = []
            self._visible_mids = set()
            self._themed_mids = set()
            self._search_results = []
            self._search_pos = -1
            self._node_xs = np.empty(0, dtype=np.float32)
            self._node_ys = np.empty(0, dtype=np.float32)
        self._set_status(f"[yellow]resolving {bot_name} graph…[/yellow]")
        if not preserve_current:
            self._render_canvas()
        self._load_graph_worker(bot_name, force_source_reload)

    @work(thread=True, exclusive=True, name="load_graph")
    def _load_graph_worker(self, bot_name: str, force_source_reload: bool) -> None:
        try:
            graph = resolve_graph(
                bot_name,
                force_source_reload=force_source_reload,
                persist=False,
            )
        except Exception as exc:
            self.app.call_from_thread(self._on_graph_failed, bot_name, str(exc))
            return
        self.app.call_from_thread(self._on_graph_ready, bot_name, graph)

    def _on_graph_failed(self, bot_name: str, error: str) -> None:
        if bot_name != self.loaded_bot:
            return
        if self.graph is not None:
            self._update_status(extra=f"refresh failed: {error}")
        else:
            self._set_status(f"[bold]failed to load graph:[/bold] {error}")
            self._render_canvas()

    def _on_graph_ready(self, bot_name: str, graph: VizGraph) -> None:
        if bot_name != self.loaded_bot:
            return
        self.graph = graph
        self._refresh_user_list()
        self._generate_viz()
        if graph.source_label == "offline checkpoint":
            self._persist_graph_worker(graph)

    @work(thread=True, exclusive=True, name="persist_graph")
    def _persist_graph_worker(self, graph: VizGraph) -> None:
        save_graph_cache(graph)

    def _set_status(self, msg: str) -> None:
        """Thread-safe status update (safe to call via call_from_thread)."""
        self.query_one("#viz-status", Static).update(msg)

    def _generate_viz(self, force_rebuild: bool = False):
        graph = self.graph
        if graph is None:
            return

        # Projection is always global. User selection changes only which nodes
        # are foregrounded/rendered, keeping stable landmarks across scopes.
        memory_ids = list(map(int, graph.memory_ids.tolist()))

        if not memory_ids:
            self._set_status("[dim]no memories to visualize[/dim]")
            self.nodes = []
            self._render_canvas()
            return

        method = self.method  # snapshot before entering thread
        bot_name = self.loaded_bot
        self._set_status(
            f"[yellow]projecting {len(memory_ids):,} memories from "
            f"{graph.source_label}…[/yellow]"
        )
        self._build_viz_worker(bot_name, method, graph, force_rebuild)

    @work(thread=True, exclusive=True, name="build_viz")
    def _build_viz_worker(
        self,
        bot_name: str,
        method: str,
        graph: VizGraph,
        force_rebuild: bool,
    ) -> None:
        """Background thread: load or build only the 2-D projection."""
        worker = get_current_worker()
        memory_ids = list(map(int, graph.memory_ids.tolist()))

        # ── cache hit ─────────────────────────────────────────────────────────
        cached = (
            None if force_rebuild
            else load_viz_cache(
                bot_name, method, memory_ids, source_key=graph.source_key
            )
        )
        if cached is not None:
            coords, raw_mags = cached
            self.app.call_from_thread(
                self._finish_viz, method, graph, coords, raw_mags, True, bot_name,
            )
            return

        # ── dimensionality reduction ──────────────────────────────────────────
        self.app.call_from_thread(
            self._set_status,
            f"[yellow]reducing dimensions ({method})…[/yellow]",
        )
        if worker.is_cancelled:
            return

        coords = reduce_dimensions_sparse(graph.sparse, method)
        raw_mags = graph.raw_magnitudes

        # ── persist to disk ───────────────────────────────────────────────────
        save_viz_cache(
            bot_name, method, memory_ids, coords, raw_mags,
            source_key=graph.source_key,
        )

        self.app.call_from_thread(
            self._finish_viz, method, graph, coords, raw_mags, False, bot_name,
        )

    def _finish_viz(
        self,
        method: str,
        graph: VizGraph,
        coords: np.ndarray,
        raw_mags: np.ndarray,
        from_cache: bool,
        bot_name: Optional[str] = None,
    ) -> None:
        """Called on the main thread once coords are ready."""
        if bot_name is not None and (
            bot_name != self.loaded_bot or method != self.method
            or self.graph is not graph
        ):
            return
        memory_ids = list(map(int, graph.memory_ids.tolist()))
        centroid   = np.mean(coords, axis=0)
        distances  = np.linalg.norm(coords - centroid, axis=1)
        dist_scale = float(np.percentile(distances, 99)) if len(distances) else 1.0
        if dist_scale <= 0:
            dist_scale = 1.0
        dist_scores = np.clip(distances / dist_scale, 0.0, 1.0)

        mag_scale = float(np.percentile(raw_mags, 99)) if len(raw_mags) else 1.0
        if mag_scale <= 0:
            mag_scale = 1.0
        mag_scores = np.clip(raw_mags / mag_scale, 0.0, 1.0)
        scores     = 0.5 * dist_scores + 0.5 * mag_scores

        self.nodes = []
        for i, mid in enumerate(memory_ids):
            self.nodes.append(VizNode(
                mid=mid, x=float(coords[i, 0]), y=float(coords[i, 1]),
                text=graph.texts[i], user_id=graph.owners[i], score=float(scores[i]),
            ))

        self.selected_idx  = 0 if self.nodes else -1
        self.connections   = []
        self.zoom          = 1.0
        self._conn_cache   = {}  # invalidate on new load
        self._visible_mids = set(memory_ids)
        if self.nodes:
            self._node_xs = coords[:, 0].astype(np.float32)
            self._node_ys = coords[:, 1].astype(np.float32)
            # Winsorized bounds stop a handful of extreme coordinates from
            # collapsing the useful map into a few terminal cells.
            if len(self.nodes) >= 20:
                self._data_min_x, self._data_max_x = map(
                    float, np.percentile(self._node_xs, [1, 99])
                )
                self._data_min_y, self._data_max_y = map(
                    float, np.percentile(self._node_ys, [1, 99])
                )
            else:
                self._data_min_x = float(self._node_xs.min())
                self._data_max_x = float(self._node_xs.max())
                self._data_min_y = float(self._node_ys.min())
                self._data_max_y = float(self._node_ys.max())
            self._node_xs = np.clip(
                self._node_xs, self._data_min_x, self._data_max_x
            )
            self._node_ys = np.clip(
                self._node_ys, self._data_min_y, self._data_max_y
            )
            self._dr_x = max(self._data_max_x - self._data_min_x, 0.001)
            self._dr_y = max(self._data_max_y - self._data_min_y, 0.001)
            self.view_cx = (self._data_min_x + self._data_max_x) / 2
            self.view_cy = (self._data_min_y + self._data_max_y) / 2
        # O(1) mid lookup for connection line drawing
        self._mid_to_node_idx = {node.mid: i for i, node in enumerate(self.nodes)}

        self._projection_cached = from_cache
        self._apply_user_scope()

    def _apply_user_scope(self, preserve_selection: bool = False) -> None:
        """Apply a user filter without changing the global projection."""
        if not self.nodes:
            self._scope_node_indices = []
            self._render_node_indices = []
            self._visible_mids = set()
            self._render_canvas()
            return

        if self.scope_user_id and self.graph:
            mids = self.graph.mids_for_user(self.scope_user_id)
            self._scope_node_indices = [
                i for i, node in enumerate(self.nodes) if node.mid in mids
            ]
        else:
            self._scope_node_indices = list(range(len(self.nodes)))

        self._visible_mids = {
            self.nodes[i].mid for i in self._scope_node_indices
        }
        self._render_node_indices = (
            list(range(len(self.nodes)))
            if self.scope_user_id and self.show_context
            else list(self._scope_node_indices)
        )
        self._conn_cache = {}

        selection_is_valid = self.selected_idx in self._scope_node_indices
        if not preserve_selection or not selection_is_valid:
            self.selected_idx = (
                self._scope_node_indices[0] if self._scope_node_indices else -1
            )
        self.connections = []
        self._apply_theme_scope()

        self._update_status()
        if self.selected_idx >= 0:
            self._show_node_details(self.nodes[self.selected_idx])
        else:
            self.query_one("#viz-detail-content", Static).update("")
            self.query_one("#viz-connections", Static).update(
                "[dim]no memories in scope[/dim]"
            )
            self._render_canvas()

    def _apply_theme_scope(self) -> None:
        self._themed_mids = (
            self.graph.themed_mids(self.scope_user_id)
            if self.graph is not None and self.show_themes else set()
        )

    def _update_status(self, extra: Optional[str] = None) -> None:
        if self.graph is None:
            return
        scope = self.scope_user_id or "all users"
        context = " + context" if self.scope_user_id and self.show_context else ""
        projection = "projection cached" if self._projection_cached else "projection built"
        state = self.graph.runtime_state
        live_state = ""
        if state.get("amygdala_response") is not None:
            live_state = f" amygdala={state['amygdala_response']}"
        theme_count = len(self.graph.themes.for_user(self.scope_user_id))
        tail = f" | {extra}" if extra else ""
        self._set_status(
            f"{len(self._scope_node_indices):,}/{len(self.nodes):,} memories "
            f"scope={scope}{context} | {self.graph.source_label} | "
            f"{projection} | themes={theme_count}{live_state}{tail}"
        )

    def _render_canvas(self):
        content = self.query_one("#viz-content", Static)
        if not self.nodes:
            content.update("[dim]Load memories to visualize[/dim]")
            self._cell_nodes = {}
            return
        if not self._scope_node_indices:
            content.update("[dim]no memories in selected user scope[/dim]")
            self._cell_nodes = {}
            return

        try:
            canvas = self.query_one("#viz-canvas")
            cw = canvas.content_size.width
            ch = canvas.content_size.height
            grid_w = max(20, cw) if cw > 0 else 80
            grid_h = max(8, ch - 1) if ch > 1 else 40
        except Exception:
            grid_w, grid_h = 80, 40
        self._grid_w = grid_w
        self._grid_h = grid_h

        data_min_x, data_max_x = self._data_min_x, self._data_max_x
        data_min_y, data_max_y = self._data_min_y, self._data_max_y
        data_range_x = self._dr_x
        data_range_y = self._dr_y

        view_range_x = data_range_x / self.zoom
        view_range_y = data_range_y / self.zoom
        if self.zoom == 1:
            view_range_x *= 1.08
            view_range_y *= 1.08
        view_min_x = self.view_cx - view_range_x / 2
        view_min_y = self.view_cy - view_range_y / 2

        pad = 1
        plot_w = max(1, grid_w - pad * 2)
        plot_h = max(1, grid_h - pad * 2)
        sub_xs = ((self._node_xs - view_min_x) / view_range_x) * (plot_w * 2)
        sub_ys = ((self._node_ys - view_min_y) / view_range_y) * (plot_h * 4)

        grid = [[" " for _ in range(grid_w)] for _ in range(grid_h)]
        active_masks = defaultdict(int)
        active_counts = defaultdict(int)
        context_cells = set()
        self._cell_nodes = defaultdict(list)
        scope_set = set(self._scope_node_indices)

        # Braille gives each terminal cell a 2x4 sub-grid. Cell membership is
        # retained separately so collisions remain navigable instead of being
        # silently overwritten by the final node drawn.
        braille_bits = (
            (0x01, 0x08),
            (0x02, 0x10),
            (0x04, 0x20),
            (0x40, 0x80),
        )
        for idx in self._render_node_indices:
            sx, sy = float(sub_xs[idx]), float(sub_ys[idx])
            if not (0 <= sx < plot_w * 2 and 0 <= sy < plot_h * 4):
                self.nodes[idx].grid_x = -1
                self.nodes[idx].grid_y = -1
                continue
            sub_x, sub_y = int(sx), int(sy)
            gx, gy = sub_x // 2 + pad, sub_y // 4 + pad
            node = self.nodes[idx]
            node.grid_x, node.grid_y = gx, gy
            cell = (gx, gy)
            if idx in scope_set:
                active_masks[cell] |= braille_bits[sub_y % 4][sub_x % 2]
                active_counts[cell] += 1
                self._cell_nodes[cell].append(idx)
            else:
                context_cells.add(cell)

        for cell in context_cells:
            x, y = cell
            grid[y][x] = "·"

        for cell, count in active_counts.items():
            x, y = cell
            stack = self._cell_nodes[cell]
            stack.sort(key=lambda idx: self.nodes[idx].score, reverse=True)
            if count == 1:
                score = self.nodes[stack[0]].score
                if self.nodes[stack[0]].mid in self._themed_mids:
                    char = "◇"
                else:
                    char = (
                        "◆" if score > 0.75 else
                        "●" if score > 0.50 else
                        "○" if score > 0.25 else "∘"
                    )
            elif count < 8:
                char = (
                    "◈" if any(self.nodes[idx].mid in self._themed_mids for idx in stack)
                    else chr(0x2800 | active_masks[cell])
                )
            elif count < 32:
                char = "▓"
            else:
                char = "█"
            grid[y][x] = char

        # Links are drawn over density, then endpoints and selection are drawn
        # last so neither can disappear beneath colliding ordinary nodes.
        if self.selected_idx >= 0 and self.connections:
            sel = self.nodes[self.selected_idx]
            for conn_mid in self.connections:
                idx = self._mid_to_node_idx.get(conn_mid, -1)
                if idx >= 0:
                    node = self.nodes[idx]
                    if (
                        0 <= sel.grid_x < grid_w and 0 <= sel.grid_y < grid_h
                        and 0 <= node.grid_x < grid_w and 0 <= node.grid_y < grid_h
                    ):
                        _draw_line(grid, sel.grid_x, sel.grid_y,
                                   node.grid_x, node.grid_y, grid_w, grid_h)

        for conn_mid in self.connections:
            idx = self._mid_to_node_idx.get(conn_mid, -1)
            if idx >= 0:
                node = self.nodes[idx]
                if 0 <= node.grid_x < grid_w and 0 <= node.grid_y < grid_h:
                    grid[node.grid_y][node.grid_x] = "◎"

        if 0 <= self.selected_idx < len(self.nodes):
            selected = self.nodes[self.selected_idx]
            if 0 <= selected.grid_x < grid_w and 0 <= selected.grid_y < grid_h:
                grid[selected.grid_y][selected.grid_x] = "◉"

        lines = ["".join(row) for row in grid]
        ext = " EXT" if self.extended_neighbors else ""
        visible_count = sum(len(stack) for stack in self._cell_nodes.values())
        collisions = visible_count - len(self._cell_nodes)
        stack = self._selected_cell_stack()
        stack_pos = (
            stack.index(self.selected_idx) + 1
            if self.selected_idx in stack else 0
        )
        stack_tag = f" cell:{stack_pos}/{len(stack)}" if len(stack) > 1 else ""
        legend = (
            f"◆hi ●mid ○lo ◇theme ◈theme-stack ▓dense █mass | ◉sel ◎conn | "
            f"z:{self.zoom:.1f} v:{visible_count}/{len(self._scope_node_indices)} "
            f"overlap:{collisions}{stack_tag}{ext} | WASD nav brackets:stack NP links"
        )
        lines.append(legend)
        content.update("\n".join(lines))

    def _selected_cell_stack(self) -> List[int]:
        if not (0 <= self.selected_idx < len(self.nodes)):
            return []
        node = self.nodes[self.selected_idx]
        return list(self._cell_nodes.get((node.grid_x, node.grid_y), []))

    def _center_on_selected(self):
        """Center viewport on selected node without re-rendering."""
        if self.nodes and 0 <= self.selected_idx < len(self.nodes):
            self.view_cx = float(self._node_xs[self.selected_idx])
            self.view_cy = float(self._node_ys[self.selected_idx])

    def _focus_selected(self):
        self._center_on_selected()
        self._render_canvas()

    def _data_ranges(self) -> Tuple[float, float]:
        if not self.nodes:
            return (1.0, 1.0)
        return (self._dr_x, self._dr_y)

    def _find_nearest(self, dx: int, dy: int):
        if not self.nodes or self.selected_idx < 0:
            return
        current = self.nodes[self.selected_idx]
        if current.grid_x < 0 or current.grid_y < 0:
            self._focus_selected()
            return
        best_idx, best_dist = -1, float("inf")

        # Navigate occupied screen cells rather than every projected node. This
        # makes directional movement deterministic even in dense collisions.
        for stack in self._cell_nodes.values():
            if not stack:
                continue
            i = stack[0]
            node = self.nodes[i]
            if (node.grid_x, node.grid_y) == (current.grid_x, current.grid_y):
                continue
            dir_x = node.grid_x - current.grid_x
            dir_y = node.grid_y - current.grid_y
            if dx != 0 and (dx * dir_x <= 0):
                continue
            if dy != 0 and (dy * dir_y <= 0):
                continue
            forward = abs(dir_x) if dx else abs(dir_y)
            perpendicular = abs(dir_y) if dx else abs(dir_x)
            dist = (
                float(np.hypot(dir_x, dir_y))
                + perpendicular * 1.5 / max(forward, 1)
            )
            if dist < best_dist:
                best_dist, best_idx = dist, i

        if best_idx >= 0:
            self.selected_idx = best_idx
            self._show_node_details(self.nodes[best_idx])  # calls _render_canvas internally

    def action_move_up(self):
        self._find_nearest(0, -1)

    def action_move_down(self):
        self._find_nearest(0, 1)

    def action_move_left(self):
        self._find_nearest(-1, 0)

    def action_move_right(self):
        self._find_nearest(1, 0)

    def action_zoom_in(self):
        if self.zoom < 32:
            self.zoom = min(32.0, self.zoom * 1.5)
            self._render_canvas()

    def action_zoom_out(self):
        if self.zoom > 1:
            self.zoom = max(1.0, self.zoom / 1.5)
            self._render_canvas()

    def action_focus_selected(self):
        self._focus_selected()

    def action_select_node(self):
        if self.nodes and 0 <= self.selected_idx < len(self.nodes):
            self._show_node_details(self.nodes[self.selected_idx], show_full=True)

    def _cycle_cell(self, step: int) -> None:
        stack = self._selected_cell_stack()
        if len(stack) < 2:
            return
        try:
            pos = stack.index(self.selected_idx)
        except ValueError:
            pos = 0
        self.selected_idx = stack[(pos + step) % len(stack)]
        self._show_node_details(self.nodes[self.selected_idx])

    def action_previous_in_cell(self):
        self._cycle_cell(-1)

    def action_next_in_cell(self):
        self._cycle_cell(1)

    def _follow_connection(self, reverse: bool = False) -> None:
        mids = list(reversed(self.connections)) if reverse else self.connections
        for mid in mids:
            idx = self._mid_to_node_idx.get(mid, -1)
            if idx in self._scope_node_indices:
                self.selected_idx = idx
                self._center_on_selected()
                self._show_node_details(self.nodes[idx])
                return

    def action_next_connection(self):
        self._follow_connection(False)

    def action_previous_connection(self):
        self._follow_connection(True)

    def action_pan_up(self):
        _, dr_y = self._data_ranges()
        self.view_cy -= dr_y / self.zoom * 0.15
        self._render_canvas()

    def action_pan_down(self):
        _, dr_y = self._data_ranges()
        self.view_cy += dr_y / self.zoom * 0.15
        self._render_canvas()

    def action_pan_left(self):
        dr_x, _ = self._data_ranges()
        self.view_cx -= dr_x / self.zoom * 0.15
        self._render_canvas()

    def action_pan_right(self):
        dr_x, _ = self._data_ranges()
        self.view_cx += dr_x / self.zoom * 0.15
        self._render_canvas()

    def on_click(self, event: Click) -> None:
        """Handle mouse clicks to select nodes."""
        if not self.nodes:
            return

        try:
            canvas = self.query_one("#viz-canvas")
        except Exception:
            return

        if not canvas.region.contains(event.screen_x, event.screen_y):
            return

        click_x = event.screen_x - canvas.region.x - 1
        click_y = event.screen_y - canvas.region.y - 1

        best_cell = None
        best_dist = float("inf")
        for cell in self._cell_nodes:
            dist = abs(cell[0] - click_x) + abs(cell[1] - click_y)
            if dist < best_dist:
                best_dist, best_cell = dist, cell

        if best_cell is not None and best_dist <= 3:
            stack = self._cell_nodes[best_cell]
            if self.selected_idx in stack and len(stack) > 1:
                pos = stack.index(self.selected_idx)
                self.selected_idx = stack[(pos + 1) % len(stack)]
            else:
                self.selected_idx = stack[0]
            self._show_node_details(self.nodes[self.selected_idx])

    def on_mouse_scroll_up(self, event) -> None:
        """Zoom in on mouse scroll up over canvas."""
        try:
            canvas = self.query_one("#viz-canvas")
            if canvas.region.contains(event.screen_x, event.screen_y):
                if self.nodes and self.zoom < 32:
                    self.zoom = min(32.0, self.zoom * 1.5)
                    self._render_canvas()
                event.stop()
        except Exception:
            pass

    def on_mouse_scroll_down(self, event) -> None:
        """Zoom out on mouse scroll down over canvas."""
        try:
            canvas = self.query_one("#viz-canvas")
            if canvas.region.contains(event.screen_x, event.screen_y):
                if self.nodes and self.zoom > 1:
                    self.zoom = max(1.0, self.zoom / 1.5)
                    self._render_canvas()
                event.stop()
        except Exception:
            pass

    def on_resize(self, event) -> None:
        """Re-render canvas when terminal resizes."""
        if self.nodes:
            self._render_canvas()

    def _show_node_details(self, node: VizNode, show_full: bool = False):
        graph = self.graph
        if graph is None:
            return
        top_k = 16 if self.extended_neighbors else 6
        cache_key = (node.mid, top_k, self.scope_user_id)
        if cache_key not in self._conn_cache:
            self._conn_cache[cache_key] = graph.connections(
                node.mid, top_k=top_k, user_id=self.scope_user_id,
            )
        connections = self._conn_cache[cache_key]
        self.connections = [c[0] for c in connections]
        self._render_canvas()

        stack = self._selected_cell_stack()
        stack_line = ""
        if len(stack) > 1 and self.selected_idx in stack:
            stack_line = (
                f"[dim]Cell stack: {stack.index(self.selected_idx) + 1}/"
                f"{len(stack)}  (brackets to cycle)[/dim]\n"
            )
        text = node.text if show_full else (
            node.text[:16000] + "..." if len(node.text) > 16000 else node.text
        )
        themes = graph.themes_for_mid(node.mid, self.scope_user_id)
        themes_line = (
            f"[dim]Themes: {', '.join(themes)}[/dim]\n" if themes else ""
        )
        header = RichText.from_markup(
            f"[bold]Memory #{node.mid}[/bold]\n"
            f"[dim]User: {node.user_id or 'unknown'}[/dim]\n"
            f"[dim]Score: {node.score:.2f}[/dim]\n"
            f"{themes_line}"
            f"{stack_line}\n"
        )
        header.append(text)
        self.query_one("#viz-detail-content", Static).update(header)

        panel = RichText()
        for i, (conn_mid, score, terms) in enumerate(connections):
            if i > 0:
                panel.append("─" * 38 + "\n", style="dim")
            panel.append(f"#{conn_mid}", style="bold")
            panel.append(f" sim={score:.2f}\n", style="dim")
            panel.append(f"shared: {', '.join(terms)}\n", style="dim")
            raw = graph.text_for_mid(conn_mid)
            if len(raw) > 400:
                raw = raw[:400] + "…"
            panel.append(raw + "\n")
        self.query_one("#viz-connections", Static).update(
            panel if connections else RichText("no connections", style="dim")
        )

    def select_mid(self, mid: int, *, focus: bool = True) -> bool:
        """Public navigation hook shared by search and future UI transports."""
        idx = self._mid_to_node_idx.get(int(mid))
        if idx is None or idx not in self._scope_node_indices:
            return False
        self.selected_idx = idx
        if focus:
            self._center_on_selected()
        self._show_node_details(self.nodes[idx])
        return True

    def current_view_payload(self) -> dict:
        """Serializable state boundary for a future web representation."""
        if self.graph is None:
            return {}
        selected = (
            self.graph.node_payload(self.nodes[self.selected_idx].mid)
            if 0 <= self.selected_idx < len(self.nodes) else None
        )
        return {
            "bot": self.graph.bot_name,
            "source": self.graph.source_label,
            "source_key": self.graph.source_key,
            "scope_user_id": self.scope_user_id,
            "projection": self.method,
            "viewport": {
                "center": [self.view_cx, self.view_cy],
                "zoom": self.zoom,
            },
            "selected": selected,
            "connections": list(self.connections),
            "runtime": dict(self.graph.runtime_state),
        }

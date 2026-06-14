export type Tab = "sources" | "frames" | "training" | "settings";

/** Sentinel values the UI uses for the character filter chip row.
 *  - "all": no character filter — the grid shows every kept frame and each
 *    thumbnail carries a small character badge.
 *  - "unsorted": map to the server's UNSORTED_FILTER_SENTINEL — frames whose
 *    slug isn't in the project's current character set (orphans from a
 *    rename/delete).
 *  Any other value is a real character slug from the project. */
export type CharacterFilter = "all" | "unsorted" | string;

class ViewStore {
  tab = $state<Tab>("frames");
  density = $state<number>(7); // columns in the frame grid (3-12)
  // Grid thumbnail scaling. false = fill the 3:4 tile (object-cover, crops);
  // true = fit the whole image inside the tile (object-contain, black bars,
  // never cropped). In-memory only, matching `density`.
  fitContain = $state<boolean>(false);
  sourceFilter = $state<string | null>(null);
  // Server-side tag filter for the frames grid. Whitespace-separated
  // substrings; tokens prefixed with `~` negate. Bound to the search input
  // in the top bar (debounced) and read by FramesTab to refresh the list.
  tagQuery = $state<string>("");

  // Active character on the Sources tab — drives the project-level ref
  // grid AND the per-video ref strips. Defaults to the project's first
  // character ("default" for legacy projects). Switching projects resets
  // this via projectsStore.load().
  activeCharacterSlug = $state<string>("default");

  // Frames-tab character filter. Independent from activeCharacterSlug so
  // the user can be browsing All while editing one character's refs on
  // the Sources tab in another tab/window.
  characterFilter = $state<CharacterFilter>("all");

  setDensity(n: number) {
    this.density = Math.max(3, Math.min(12, n));
  }
}

export const viewStore = new ViewStore();

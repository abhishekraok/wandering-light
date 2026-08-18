import "./styles.css";

import { ApiError, api } from "./api";
import {
  clear,
  clampNumber,
  compactId,
  compactType,
  debounce,
  formatCount,
  previewSerialized,
  requiredElement,
  setHidden,
  textElement,
} from "./dom";
import { TrajectoryGraphCanvas, type GraphSelection } from "./graph";
import { serializedTypedListType } from "./graph-model";
import type {
  BasisFunction,
  CandidateTask,
  CorpusSource,
  ExpansionGraphRequest,
  ExpansionGraphResponse,
  FacetsResponse,
  GraphPayload,
  TaskDetail,
  TaskQuery,
  TaskSummary,
  WitnessGraphResponse,
} from "./types";

type GraphMode = "witness" | "expand";
type NoticeKind = "error" | "warning" | "info";

const TASK_PAGE_SIZE = 50;
const TASK_ACCUMULATION_CAP = 400;
const CANDIDATE_PAGE_SIZE = 50;
const GRAPH_NODE_CAP = 900;
const GRAPH_EDGE_CAP = 2_000;
const LAST_SOURCE_KEY = "wandering-light:last-graph-source";

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiError || error instanceof Error) return error.message;
  return String(error);
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function appendDefinition(container: HTMLElement, label: string, value: string): void {
  const row = document.createElement("div");
  row.className = "definition-row";
  row.append(textElement("dt", label), textElement("dd", value));
  container.append(row);
}

class TrajectoryGraphApp {
  private readonly sourceSelect = requiredElement<HTMLSelectElement>("#source-select");
  private readonly sourceCard = requiredElement<HTMLElement>("#source-card");
  private readonly sourceAction = requiredElement<HTMLButtonElement>("#source-action");
  private readonly headerContext = requiredElement<HTMLElement>("#header-context");
  private readonly headerStatusDot = requiredElement<HTMLElement>("#header-status-dot");
  private readonly filterForm = requiredElement<HTMLFormElement>("#filter-form");
  private readonly taskSearch = requiredElement<HTMLInputElement>("#task-search");
  private readonly splitFilter = requiredElement<HTMLSelectElement>("#split-filter");
  private readonly roleFilter = requiredElement<HTMLSelectElement>("#role-filter");
  private readonly minDistance = requiredElement<HTMLInputElement>("#min-distance");
  private readonly maxDistance = requiredElement<HTMLInputElement>("#max-distance");
  private readonly functionFilter = requiredElement<HTMLSelectElement>("#function-filter");
  private readonly taskCount = requiredElement<HTMLElement>("#task-count");
  private readonly taskFilterState = requiredElement<HTMLElement>("#task-filter-state");
  private readonly taskList = requiredElement<HTMLElement>("#task-list");
  private readonly loadMoreTasksButton = requiredElement<HTMLButtonElement>("#load-more-tasks");
  private readonly selectedTaskPanel = requiredElement<HTMLElement>("#selected-task-panel");
  private readonly witnessControls = requiredElement<HTMLElement>("#witness-controls");
  private readonly expansionControls = requiredElement<HTMLElement>("#expansion-controls");
  private readonly maxRecordsField = requiredElement<HTMLElement>("#max-records-field");
  private readonly maxRecordsInput = requiredElement<HTMLInputElement>("#max-records");
  private readonly maxRecordsOutput = requiredElement<HTMLOutputElement>("#max-records-output");
  private readonly expansionInput = requiredElement<HTMLTextAreaElement>("#expansion-input");
  private readonly paletteSearch = requiredElement<HTMLInputElement>("#palette-search");
  private readonly paletteCount = requiredElement<HTMLElement>("#palette-count");
  private readonly functionPalette = requiredElement<HTMLElement>("#function-palette");
  private readonly maxDepth = requiredElement<HTMLInputElement>("#max-depth");
  private readonly maxStates = requiredElement<HTMLInputElement>("#max-states");
  private readonly maxTransitions = requiredElement<HTMLInputElement>("#max-transitions");
  private readonly includeSelfLoops = requiredElement<HTMLInputElement>("#include-self-loops");
  private readonly inspectorSection = requiredElement<HTMLElement>(".inspector-section");
  private readonly inspector = requiredElement<HTMLElement>("#inspector-content");
  private readonly candidateSection = requiredElement<HTMLElement>("#candidate-section");
  private readonly candidateCount = requiredElement<HTMLElement>("#candidate-count");
  private readonly candidateList = requiredElement<HTMLElement>("#candidate-list");
  private readonly moreCandidates = requiredElement<HTMLButtonElement>("#more-candidates");
  private readonly graphSummary = requiredElement<HTMLElement>("#graph-summary");
  private readonly canvasEmpty = requiredElement<HTMLElement>("#canvas-empty");
  private readonly canvasEmptyTitle = requiredElement<HTMLElement>("#canvas-empty-title");
  private readonly canvasEmptyCopy = requiredElement<HTMLElement>("#canvas-empty-copy");
  private readonly canvasHint = requiredElement<HTMLElement>("#canvas-hint");
  private readonly graphLoading = requiredElement<HTMLElement>("#graph-loading");
  private readonly graphLoadingTitle = requiredElement<HTMLElement>("#graph-loading-title");
  private readonly graphLoadingDetail = requiredElement<HTMLElement>("#graph-loading-detail");
  private readonly noticeStack = requiredElement<HTMLElement>("#notice-stack");
  private readonly fitGraphButton = requiredElement<HTMLButtonElement>("#fit-graph");
  private readonly resetLayoutButton = requiredElement<HTMLButtonElement>("#reset-layout");
  private readonly clearPathButton = requiredElement<HTMLButtonElement>("#clear-path");
  private readonly toggleBrowserButton = requiredElement<HTMLButtonElement>("#toggle-browser");
  private readonly toggleDetailsButton = requiredElement<HTMLButtonElement>("#toggle-details");
  private readonly graph: TrajectoryGraphCanvas;

  private sources: CorpusSource[] = [];
  private selectedSource: CorpusSource | null = null;
  private facets: FacetsResponse | null = null;
  private tasks: TaskSummary[] = [];
  private taskCursor: string | null = null;
  private taskTotal = 0;
  private taskCapReached = false;
  private selectedTask: TaskDetail | null = null;
  private selectedFunctionIds = new Set<string>();
  private candidates: CandidateTask[] = [];
  private visibleCandidateCount = CANDIDATE_PAGE_SIZE;
  private sourceGeneration = 0;
  private taskController: AbortController | null = null;
  private detailController: AbortController | null = null;
  private graphController: AbortController | null = null;
  private sourceActionPending = false;
  private activeMode: GraphMode = "witness";
  private displayedGraphMode: GraphMode | null = null;

  constructor() {
    this.graph = new TrajectoryGraphCanvas(
      requiredElement<HTMLElement>("#graph-canvas"),
      (selection) => {
        this.renderInspector(selection);
        if (selection.kind !== "none" && this.toolsAreDrawer()) {
          this.openMobileRail("tools");
          window.requestAnimationFrame(() =>
            this.inspectorSection.scrollIntoView({ block: "start" }),
          );
        }
      },
    );
    this.bindEvents();
  }

  async start(): Promise<void> {
    this.setHeader("loading", "Connecting to corpus index…");
    try {
      const response = await api.sources();
      this.sources = response.sources;
      this.renderSourceSelect();
      for (const discoveryError of response.errors.slice(0, 3)) {
        this.showNotice(`${discoveryError.location}: ${discoveryError.message}`, "warning", false);
      }
      if (this.sources.length === 0) {
        this.renderNoSources();
        return;
      }
      const remembered = this.readLastSource();
      const initial = this.sources.find((source) => source.id === remembered) ?? this.sources[0];
      this.sourceSelect.value = initial.id;
      await this.activateSource(initial.id);
    } catch (error) {
      this.setHeader("error", "API unavailable");
      this.renderFatalError(
        `Could not reach the trajectory graph API. ${errorMessage(error)}`,
      );
    }
  }

  private bindEvents(): void {
    this.sourceSelect.addEventListener("change", () => void this.activateSource(this.sourceSelect.value));
    this.sourceAction.addEventListener("click", () => void this.handleSourceAction());

    this.filterForm.addEventListener("submit", (event) => event.preventDefault());
    const reloadTasks = () => void this.loadTasks(true);
    this.splitFilter.addEventListener("change", reloadTasks);
    this.roleFilter.addEventListener("change", reloadTasks);
    this.functionFilter.addEventListener("change", reloadTasks);
    this.minDistance.addEventListener("change", reloadTasks);
    this.maxDistance.addEventListener("change", reloadTasks);
    this.taskSearch.addEventListener("input", debounce(reloadTasks, 260));
    requiredElement<HTMLButtonElement>("#reset-filters").addEventListener("click", () => {
      this.filterForm.reset();
      void this.loadTasks(true);
    });
    this.loadMoreTasksButton.addEventListener("click", () => void this.loadTasks(false));

    requiredElement<HTMLButtonElement>("#mode-witness").addEventListener("click", () => {
      this.setMode("witness");
      if (this.selectedTask) void this.loadWitnessGraph();
    });
    requiredElement<HTMLButtonElement>("#mode-expand").addEventListener("click", () => {
      this.setMode("expand");
      this.openMobileRail("tools");
    });
    document.querySelectorAll<HTMLInputElement>('input[name="witness-scope"]').forEach((input) => {
      input.addEventListener("change", () => {
        setHidden(this.maxRecordsField, this.witnessScope() !== "root");
        if (this.selectedTask) void this.loadWitnessGraph();
      });
    });
    this.maxRecordsInput.addEventListener("input", () => {
      this.maxRecordsOutput.value = this.maxRecordsInput.value;
    });
    requiredElement<HTMLButtonElement>("#reload-witness").addEventListener("click", () => void this.loadWitnessGraph());

    this.paletteSearch.addEventListener("input", () => this.renderFunctionPalette());
    requiredElement<HTMLButtonElement>("#palette-witness").addEventListener("click", () => {
      this.selectWitnessFunctions();
      this.renderFunctionPalette();
    });
    requiredElement<HTMLButtonElement>("#palette-compatible").addEventListener("click", () => {
      this.selectCompatibleFunctions();
      this.renderFunctionPalette();
    });
    requiredElement<HTMLButtonElement>("#palette-clear").addEventListener("click", () => {
      this.selectedFunctionIds.clear();
      this.renderFunctionPalette();
    });
    requiredElement<HTMLButtonElement>("#run-expansion").addEventListener("click", () => void this.runExpansion());

    this.fitGraphButton.addEventListener("click", () => this.graph.fit());
    this.resetLayoutButton.addEventListener("click", () => this.graph.resetLayout());
    this.clearPathButton.addEventListener("click", () => this.graph.clearPath());
    this.moreCandidates.addEventListener("click", () => {
      this.visibleCandidateCount += CANDIDATE_PAGE_SIZE;
      this.renderCandidates();
    });

    const helpDialog = requiredElement<HTMLDialogElement>("#interaction-help");
    requiredElement<HTMLButtonElement>("#open-help").addEventListener("click", () => helpDialog.showModal());
    requiredElement<HTMLButtonElement>("#close-help").addEventListener("click", () => helpDialog.close());
    helpDialog.addEventListener("click", (event) => {
      if (event.target === helpDialog) helpDialog.close();
    });

    requiredElement<HTMLButtonElement>("#toggle-browser").addEventListener("click", () => this.openMobileRail("browser"));
    requiredElement<HTMLButtonElement>("#toggle-details").addEventListener("click", () => this.openMobileRail("tools"));
    requiredElement<HTMLButtonElement>("#rail-scrim").addEventListener("click", () => this.closeMobileRails());
    document.querySelectorAll<HTMLButtonElement>("[data-close-rail]").forEach((button) => {
      button.addEventListener("click", () => this.closeMobileRails());
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") this.closeMobileRails();
    });
  }

  private renderSourceSelect(): void {
    clear(this.sourceSelect);
    for (const source of this.sources) {
      const option = document.createElement("option");
      option.value = source.id;
      option.textContent = source.name;
      this.sourceSelect.append(option);
    }
    this.sourceSelect.disabled = this.sources.length === 0;
  }

  private async activateSource(sourceId: string): Promise<void> {
    const source = this.sources.find((item) => item.id === sourceId);
    if (!source) return;
    this.sourceGeneration += 1;
    const generation = this.sourceGeneration;
    this.abortRequests();
    this.selectedSource = source;
    this.facets = null;
    this.tasks = [];
    this.taskCursor = null;
    this.taskTotal = 0;
    this.taskCapReached = false;
    this.selectedTask = null;
    this.candidates = [];
    this.selectedFunctionIds.clear();
    this.renderSourceCard();
    this.renderTaskList();
    this.renderSelectedTask();
    this.resetWorkspaceGraph();
    setHidden(this.witnessControls, true);
    setHidden(this.expansionControls, true);
    setHidden(this.candidateSection, true);
    this.writeLastSource(source.id);

    if (!source.ready) {
      this.setHeader("warning", `${source.name} needs its corpus payload`);
      return;
    }
    if (source.index.status === "ready") {
      await this.loadReadySource(generation);
      return;
    }
    await this.startAndPollIndex(generation);
  }

  private async startAndPollIndex(generation: number): Promise<void> {
    if (!this.selectedSource || generation !== this.sourceGeneration) return;
    const sourceId = this.selectedSource.id;
    try {
      this.setHeader("loading", `Preparing ${this.selectedSource.name}…`);
      await api.startIndex(sourceId);
      while (generation === this.sourceGeneration) {
        const response = await api.sources();
        if (generation !== this.sourceGeneration) return;
        const updated = response.sources.find((source) => source.id === sourceId);
        if (!updated) throw new Error("The selected corpus disappeared during indexing.");
        this.sources = response.sources;
        this.selectedSource = updated;
        this.renderSourceCard();
        if (updated.index.status === "ready") {
          await this.loadReadySource(generation);
          return;
        }
        if (updated.index.status === "error") {
          this.setHeader("error", `Index failed for ${updated.name}`);
          return;
        }
        this.setHeader(
          "loading",
          `${updated.index.message ?? "Indexing"} · ${formatCount(updated.index.records_indexed)} records`,
        );
        await delay(550);
      }
    } catch (error) {
      if (generation !== this.sourceGeneration || isAbort(error)) return;
      this.setHeader("error", "Corpus index unavailable");
      this.showNotice(errorMessage(error), "error");
      this.renderSourceCard();
    }
  }

  private async loadReadySource(generation: number): Promise<void> {
    if (!this.selectedSource || generation !== this.sourceGeneration) return;
    const sourceId = this.selectedSource.id;
    this.setHeader("ready", `${this.selectedSource.name} · index ready`);
    try {
      const facets = await api.facets(sourceId);
      if (generation !== this.sourceGeneration) return;
      this.facets = facets;
      this.renderFacets();
      this.renderSourceCard();
      await this.loadTasks(true);
    } catch (error) {
      if (generation !== this.sourceGeneration || isAbort(error)) return;
      this.showNotice(`Could not load corpus facets: ${errorMessage(error)}`, "error");
    }
  }

  private async handleSourceAction(): Promise<void> {
    if (!this.selectedSource || this.sourceActionPending) return;
    this.sourceActionPending = true;
    this.sourceAction.disabled = true;
    const sourceId = this.selectedSource.id;
    const generation = this.sourceGeneration;
    try {
      if (!this.selectedSource.ready) {
        this.sourceAction.textContent = "Fetching and verifying…";
        await api.fetchSource(sourceId);
        if (generation !== this.sourceGeneration) return;
        const response = await api.sources();
        if (generation !== this.sourceGeneration) return;
        this.sources = response.sources;
        this.renderSourceSelect();
        this.sourceSelect.value = sourceId;
        await this.activateSource(sourceId);
      } else {
        await this.startAndPollIndex(generation);
      }
    } catch (error) {
      if (generation === this.sourceGeneration) this.showNotice(errorMessage(error), "error");
    } finally {
      this.sourceActionPending = false;
      this.sourceAction.disabled = false;
      if (generation === this.sourceGeneration) this.renderSourceCard();
    }
  }

  private renderSourceCard(): void {
    clear(this.sourceCard);
    if (!this.selectedSource) return;
    const source = this.selectedSource;
    const heading = document.createElement("div");
    heading.className = "source-card-heading";
    heading.append(textElement("strong", source.name));
    const status = textElement("span", source.index.status, `status-pill ${source.index.status}`);
    heading.append(status);
    this.sourceCard.append(heading);

    const details = document.createElement("dl");
    details.className = "source-details";
    appendDefinition(details, "Expected tasks", formatCount(source.expected_records));
    appendDefinition(details, "Basis", source.basis_set_id ?? "Legacy / inferred");
    if (this.facets) {
      appendDefinition(details, "Indexed tasks", formatCount(this.facets.stats.records));
      appendDefinition(details, "Certified", formatCount(this.facets.stats.certified_records));
    } else if (source.index.status === "indexing") {
      appendDefinition(details, "Indexed so far", formatCount(source.index.records_indexed));
    }
    this.sourceCard.append(details);

    const progress = document.createElement("div");
    progress.className = "index-progress";
    const fill = document.createElement("span");
    const expected = source.expected_records ?? 0;
    const indexed = source.index.records_indexed ?? 0;
    fill.style.width = expected > 0 ? `${Math.min(100, (indexed / expected) * 100)}%` : source.index.status === "ready" ? "100%" : "18%";
    progress.append(fill);
    if (source.index.status === "indexing") this.sourceCard.append(progress);

    let action = "";
    if (!source.ready && source.fetchable) action = "Fetch corpus payload";
    else if (!source.ready) action = "Payload unavailable";
    else if (source.index.status === "error") action = "Retry indexing";
    else if (source.index.status === "idle") action = "Build index";
    setHidden(this.sourceAction, action === "" || source.index.status === "indexing");
    this.sourceAction.textContent = action;
    this.sourceAction.disabled = (!source.ready && !source.fetchable) || this.sourceActionPending;
  }

  private renderFacets(): void {
    if (!this.facets) return;
    const selectedSplit = this.splitFilter.value;
    clear(this.splitFilter);
    const allSplits = document.createElement("option");
    allSplits.value = "";
    allSplits.textContent = "All splits";
    this.splitFilter.append(allSplits);
    for (const split of this.facets.splits) {
      const option = document.createElement("option");
      option.value = split;
      option.textContent = split;
      this.splitFilter.append(option);
    }
    this.splitFilter.value = selectedSplit;

    const selectedFunction = this.functionFilter.value;
    clear(this.functionFilter);
    const anyFunction = document.createElement("option");
    anyFunction.value = "";
    anyFunction.textContent = "Any function";
    this.functionFilter.append(anyFunction);
    for (const functionItem of this.facets.functions) {
      const option = document.createElement("option");
      option.value = functionItem.function_key;
      option.textContent = `${functionItem.function_name} · ${formatCount(functionItem.records)}`;
      this.functionFilter.append(option);
    }
    this.functionFilter.value = selectedFunction;

    const distances = this.facets.distance_counts.map((item) => item.value);
    if (distances.length > 0) {
      const minimum = Math.min(...distances);
      const maximum = Math.max(...distances);
      this.minDistance.min = String(minimum);
      this.minDistance.max = String(maximum);
      this.maxDistance.min = String(minimum);
      this.maxDistance.max = String(maximum);
    }
  }

  private currentTaskQuery(cursor?: string): TaskQuery {
    const minimum = this.minDistance.value === "" ? undefined : Number(this.minDistance.value);
    const maximum = this.maxDistance.value === "" ? undefined : Number(this.maxDistance.value);
    return {
      split: this.splitFilter.value || undefined,
      minDistance: minimum,
      maxDistance: maximum,
      functionKey: this.functionFilter.value || undefined,
      functionRole: this.roleFilter.value || undefined,
      taskPrefix: this.taskSearch.value.trim() || undefined,
      cursor,
      limit: TASK_PAGE_SIZE,
    };
  }

  private async loadTasks(reset: boolean): Promise<void> {
    if (!this.selectedSource || this.selectedSource.index.status !== "ready") return;
    if (!reset && !this.taskCursor) return;
    this.taskController?.abort();
    const controller = new AbortController();
    this.taskController = controller;
    const sourceId = this.selectedSource.id;
    const cursor = reset ? undefined : this.taskCursor ?? undefined;
    if (reset) {
      this.tasks = [];
      this.taskCursor = null;
      this.taskCapReached = false;
      this.renderTaskLoading();
    } else {
      this.loadMoreTasksButton.disabled = true;
      this.loadMoreTasksButton.textContent = "Loading…";
    }
    try {
      const response = await api.tasks(sourceId, this.currentTaskQuery(cursor), controller.signal);
      if (controller.signal.aborted || this.selectedSource?.id !== sourceId) return;
      const accumulated = reset ? response.items : [...this.tasks, ...response.items];
      this.taskCapReached = accumulated.length >= TASK_ACCUMULATION_CAP && response.next_cursor !== null;
      this.tasks = accumulated.slice(0, TASK_ACCUMULATION_CAP);
      this.taskCursor = this.taskCapReached ? null : response.next_cursor;
      this.taskTotal = response.total;
      this.renderTaskList();
      if (reset && !this.selectedTask && response.items[0]) {
        await this.selectTask(response.items[0].row_id);
      }
    } catch (error) {
      if (isAbort(error)) return;
      this.renderTaskError(errorMessage(error));
    } finally {
      if (this.taskController === controller) {
        this.taskController = null;
        this.loadMoreTasksButton.disabled = false;
        this.loadMoreTasksButton.textContent = "Load more tasks";
      }
    }
  }

  private renderTaskLoading(): void {
    clear(this.taskList);
    for (let index = 0; index < 5; index += 1) {
      const skeleton = document.createElement("div");
      skeleton.className = "task-card task-skeleton";
      skeleton.append(textElement("span", "", "skeleton-line"), textElement("span", "", "skeleton-line short"));
      this.taskList.append(skeleton);
    }
    this.taskCount.textContent = "Loading tasks…";
    this.taskFilterState.textContent = "";
    setHidden(this.loadMoreTasksButton, true);
  }

  private renderTaskList(): void {
    clear(this.taskList);
    if (this.tasks.length === 0) {
      const empty = document.createElement("div");
      empty.className = "list-empty";
      empty.append(textElement("strong", this.selectedSource ? "No matching tasks" : "Choose a corpus"));
      empty.append(textElement("p", this.selectedSource ? "Try widening the filters above." : "Sources will appear when the API is ready."));
      this.taskList.append(empty);
    }
    for (const task of this.tasks) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "task-card";
      button.classList.toggle("selected", this.selectedTask?.row_id === task.row_id);
      button.dataset.rowId = String(task.row_id);
      button.setAttribute("aria-pressed", String(this.selectedTask?.row_id === task.row_id));

      const top = document.createElement("span");
      top.className = "task-card-top";
      top.append(
        textElement("span", task.distance === null ? "d?" : `d${task.distance}`, "distance-badge"),
        textElement("span", `${compactType(task.input_type)} → ${compactType(task.output_type)}`, "type-pair"),
        textElement("span", task.split, "split-tag"),
      );
      const io = textElement("span", `${task.input_preview}  →  ${task.output_preview}`, "task-io");
      const witness = textElement(
        "span",
        task.witness_function_names.length > 0 ? task.witness_function_names.join(" → ") : "No witness functions",
        "task-witness",
      );
      button.append(top, io, witness);
      button.addEventListener("click", () => void this.selectTask(task.row_id));
      this.taskList.append(button);
    }
    this.taskCount.textContent = `${formatCount(this.taskTotal)} matching task${this.taskTotal === 1 ? "" : "s"}`;
    this.taskFilterState.textContent = this.taskCapReached
      ? `${formatCount(this.tasks.length)} shown · refine filters`
      : this.tasks.length < this.taskTotal
        ? `${formatCount(this.tasks.length)} shown`
        : "All shown";
    setHidden(this.loadMoreTasksButton, !this.taskCursor);
  }

  private renderTaskError(message: string): void {
    clear(this.taskList);
    const empty = document.createElement("div");
    empty.className = "list-empty error-state";
    empty.append(textElement("strong", "Tasks could not be loaded"), textElement("p", message));
    this.taskList.append(empty);
    this.taskCount.textContent = "Task query failed";
    setHidden(this.loadMoreTasksButton, true);
  }

  private async selectTask(rowId: number): Promise<void> {
    if (!this.selectedSource) return;
    this.detailController?.abort();
    this.graphController?.abort();
    const controller = new AbortController();
    this.detailController = controller;
    const sourceId = this.selectedSource.id;
    if (this.selectedTask?.row_id !== rowId) this.resetWorkspaceGraph();
    this.markSelectedTask(rowId);
    this.selectedTaskPanel.classList.add("loading-panel");
    try {
      const detail = await api.task(sourceId, rowId, controller.signal);
      if (controller.signal.aborted || this.selectedSource?.id !== sourceId) return;
      this.selectedTask = detail;
      this.expansionInput.value = detail.input;
      this.paletteSearch.value = "";
      this.selectWitnessFunctions();
      if (this.selectedFunctionIds.size === 0) this.selectCompatibleFunctions(10);
      this.updateRootScopeAvailability();
      this.renderSelectedTask();
      this.renderFunctionPalette();
      this.setMode("witness");
      this.renderTaskList();
      this.closeMobileRails();
      await this.loadWitnessGraph();
    } catch (error) {
      if (isAbort(error)) return;
      this.showNotice(`Could not load task: ${errorMessage(error)}`, "error");
    } finally {
      if (this.detailController === controller) {
        this.selectedTaskPanel.classList.remove("loading-panel");
        this.detailController = null;
      }
    }
  }

  private markSelectedTask(rowId: number): void {
    this.taskList.querySelectorAll<HTMLButtonElement>(".task-card").forEach((button) => {
      const selected = Number(button.dataset.rowId) === rowId;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
  }

  private renderSelectedTask(): void {
    clear(this.selectedTaskPanel);
    if (!this.selectedTask) {
      this.selectedTaskPanel.append(textElement("p", "Select a corpus task to begin.", "muted-copy"));
      return;
    }
    const task = this.selectedTask;
    const heading = document.createElement("div");
    heading.className = "selected-task-heading";
    const titleGroup = document.createElement("div");
    titleGroup.append(textElement("span", task.distance === null ? "d?" : `d${task.distance}`, "distance-badge large"));
    titleGroup.append(textElement("strong", `${compactType(task.input_type)} → ${compactType(task.output_type)}`));
    heading.append(titleGroup, textElement("code", compactId(task.task_id, 12)));

    const input = document.createElement("div");
    input.className = "task-value";
    input.append(textElement("span", "Input"), textElement("code", previewSerialized(task.input)));
    const output = document.createElement("div");
    output.className = "task-value";
    output.append(textElement("span", "Target"), textElement("code", previewSerialized(task.output)));

    const witness = document.createElement("div");
    witness.className = "witness-chain";
    witness.append(textElement("span", "Witness"));
    const chain = document.createElement("div");
    for (const name of task.witness_function_names) chain.append(textElement("code", name));
    if (task.witness_function_names.length === 0) chain.append(textElement("em", "No stored path"));
    witness.append(chain);
    this.selectedTaskPanel.append(heading, input, output, witness);
  }

  private setMode(mode: GraphMode): void {
    const changed = this.activeMode !== mode;
    this.activeMode = mode;
    requiredElement<HTMLButtonElement>("#mode-witness").classList.toggle("active", mode === "witness");
    requiredElement<HTMLButtonElement>("#mode-expand").classList.toggle("active", mode === "expand");
    setHidden(this.witnessControls, !this.selectedTask || mode !== "witness");
    setHidden(this.expansionControls, !this.selectedTask || mode !== "expand");
    setHidden(this.candidateSection, mode !== "expand" || this.candidates.length === 0);
    if (
      changed &&
      mode === "expand" &&
      this.selectedTask !== null &&
      this.displayedGraphMode !== "expand"
    ) {
      this.showExpansionPrompt();
    }
  }

  private witnessScope(): "selected" | "root" {
    const checked = document.querySelector<HTMLInputElement>('input[name="witness-scope"]:checked');
    return checked?.value === "root" ? "root" : "selected";
  }

  private updateRootScopeAvailability(): void {
    const rootInput = document.querySelector<HTMLInputElement>('input[name="witness-scope"][value="root"]');
    const selectedInput = document.querySelector<HTMLInputElement>('input[name="witness-scope"][value="selected"]');
    if (!rootInput || !selectedInput) return;
    const available = this.selectedTask?.root_index !== null && this.selectedTask?.root_index !== undefined;
    rootInput.disabled = !available;
    rootInput.closest("label")?.classList.toggle("disabled", !available);
    rootInput.closest("label")?.setAttribute(
      "title",
      available ? "Merge stored witnesses sampled from this root" : "This record has no root grouping",
    );
    if (!available && rootInput.checked) selectedInput.checked = true;
    setHidden(this.maxRecordsField, this.witnessScope() !== "root");
  }

  private async loadWitnessGraph(): Promise<void> {
    if (!this.selectedSource || !this.selectedTask) return;
    this.graphController?.abort();
    const controller = new AbortController();
    this.graphController = controller;
    const sourceId = this.selectedSource.id;
    const task = this.selectedTask;
    const scope = this.witnessScope();
    const maxRecords = clampNumber(this.maxRecordsInput, 250);
    this.setGraphLoading(true, scope === "root" ? "Merging root witnesses" : "Replaying witness", "Executing registered functions and deduplicating states…");
    try {
      const response = await api.witnessGraph(
        sourceId,
        {
          row_id: task.row_id,
          scope,
          max_records: maxRecords,
          max_nodes: GRAPH_NODE_CAP,
          max_edges: GRAPH_EDGE_CAP,
        },
        controller.signal,
      );
      if (controller.signal.aborted || this.selectedTask?.row_id !== task.row_id) return;
      const graphKey = `witness:${sourceId}:${task.row_id}:${scope}:${maxRecords}`;
      this.showGraph(graphKey, response.graph, "witness");
      this.renderWitnessSummary(response);
      this.candidates = [];
      setHidden(this.candidateSection, true);
      for (const projectionError of response.errors.slice(0, 3)) {
        this.showNotice(projectionError, "warning");
      }
    } catch (error) {
      if (isAbort(error)) return;
      this.showNotice(`Witness graph failed: ${errorMessage(error)}`, "error");
    } finally {
      if (this.graphController === controller) {
        this.graphController = null;
        this.setGraphLoading(false);
      }
    }
  }

  private selectedPaletteFunctions(): Array<BasisFunction & { id: string }> {
    return (this.selectedTask?.basis.functions ?? []).filter(
      (functionItem): functionItem is BasisFunction & { id: string } =>
        functionItem.id !== null && this.selectedFunctionIds.has(functionItem.id),
    );
  }

  private selectWitnessFunctions(): void {
    this.selectedFunctionIds.clear();
    if (!this.selectedTask) return;
    const ids = this.selectedTask.witness_function_ids.filter(Boolean);
    if (ids.length > 0) {
      ids.forEach((id) => this.selectedFunctionIds.add(id));
      return;
    }
    const names = new Set(this.selectedTask.witness_function_names);
    for (const functionItem of this.selectedTask.basis.functions) {
      if (functionItem.id && names.has(functionItem.name)) this.selectedFunctionIds.add(functionItem.id);
    }
  }

  private selectCompatibleFunctions(limit?: number): boolean {
    if (!this.selectedTask) return false;
    const rootType = serializedTypedListType(this.expansionInput.value);
    if (rootType === null) {
      this.showNotice(
        "Enter valid built-in TypedList JSON before selecting compatible functions.",
        "warning",
      );
      return false;
    }
    this.selectedFunctionIds.clear();
    const compatible = this.selectedTask.basis.functions.filter(
      (functionItem) => functionItem.id && functionItem.input_type === rootType,
    );
    for (const functionItem of limit === undefined ? compatible : compatible.slice(0, limit)) {
      if (functionItem.id) this.selectedFunctionIds.add(functionItem.id);
    }
    return true;
  }

  private renderFunctionPalette(): void {
    clear(this.functionPalette);
    if (!this.selectedTask) return;
    const query = this.paletteSearch.value.trim().toLocaleLowerCase();
    const functions = this.selectedTask.basis.functions.filter((functionItem) =>
      `${functionItem.name} ${functionItem.input_type} ${functionItem.output_type}`.toLocaleLowerCase().includes(query),
    );
    for (const functionItem of functions) {
      const label = document.createElement("label");
      label.className = "function-option";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.disabled = functionItem.id === null;
      checkbox.checked = functionItem.id !== null && this.selectedFunctionIds.has(functionItem.id);
      checkbox.addEventListener("change", () => {
        if (!functionItem.id) return;
        if (checkbox.checked) this.selectedFunctionIds.add(functionItem.id);
        else this.selectedFunctionIds.delete(functionItem.id);
        this.updatePaletteCount();
      });
      const copy = document.createElement("span");
      copy.append(textElement("strong", functionItem.name));
      copy.append(textElement("small", `${compactType(functionItem.input_type)} → ${compactType(functionItem.output_type)}`));
      label.append(checkbox, copy);
      this.functionPalette.append(label);
    }
    if (functions.length === 0) this.functionPalette.append(textElement("p", "No functions match this filter.", "muted-copy palette-empty"));
    this.updatePaletteCount();
  }

  private updatePaletteCount(): void {
    this.paletteCount.textContent = `${this.selectedFunctionIds.size} selected`;
  }

  private async runExpansion(): Promise<void> {
    if (!this.selectedSource || !this.selectedTask) return;
    const functionIds = this.selectedPaletteFunctions().map((functionItem) => functionItem.id);
    if (functionIds.length === 0) {
      this.showNotice("Select at least one registered function before expanding.", "warning");
      return;
    }
    this.graphController?.abort();
    const controller = new AbortController();
    this.graphController = controller;
    const sourceId = this.selectedSource.id;
    const task = this.selectedTask;
    const body: ExpansionGraphRequest = {
      row_id: task.row_id,
      input_serialized: this.expansionInput.value,
      function_ids: functionIds,
      max_depth: clampNumber(this.maxDepth, 2),
      max_states: clampNumber(this.maxStates, 250),
      max_transitions: clampNumber(this.maxTransitions, 2500),
      include_self_loops: this.includeSelfLoops.checked,
      max_nodes: GRAPH_NODE_CAP,
      max_edges: GRAPH_EDGE_CAP,
    };
    this.setGraphLoading(true, "Expanding local graph", `Applying ${functionIds.length} functions within hard budgets…`);
    try {
      const response = await api.expansionGraph(sourceId, body, controller.signal);
      if (controller.signal.aborted || this.selectedTask?.row_id !== task.row_id) return;
      const graphKey = `expand:${sourceId}:${JSON.stringify(body)}`;
      this.showGraph(graphKey, response.graph, "expand");
      this.renderExpansionSummary(response);
      this.candidates = response.tasks;
      this.visibleCandidateCount = CANDIDATE_PAGE_SIZE;
      this.renderCandidates();
      setHidden(this.candidateSection, this.candidates.length === 0);
      if (response.stop_reason) {
        this.showNotice(
          `Expansion stopped at ${response.stop_reason.replaceAll("_", " ")}; distances beyond certified depth ${response.certified_depth} are provisional.`,
          "warning",
          false,
        );
      }
      this.closeMobileRails();
    } catch (error) {
      if (isAbort(error)) return;
      this.showNotice(`Expansion failed: ${errorMessage(error)}`, "error");
    } finally {
      if (this.graphController === controller) {
        this.graphController = null;
        this.setGraphLoading(false);
      }
    }
  }

  private showGraph(graphKey: string, payload: GraphPayload, mode: GraphMode): void {
    this.graph.load(graphKey, payload);
    this.displayedGraphMode = mode;
    setHidden(this.canvasEmpty, true);
    setHidden(this.graphSummary, false);
    this.canvasHint.textContent =
      mode === "expand"
        ? "Click a state to trace it · choose a candidate in Tools · drag or zoom freely"
        : "Drag nodes · scroll to zoom · click a state to trace its shortest path";
    this.fitGraphButton.disabled = false;
    this.resetLayoutButton.disabled = false;
    this.clearPathButton.disabled = false;
  }

  private renderWitnessSummary(response: WitnessGraphResponse): void {
    this.renderGraphSummary(response.graph, [
      ["Projection", "Stored"],
      ["Witnesses", formatCount(response.processed_records)],
      ["Skipped", formatCount(response.skipped_records)],
    ]);
  }

  private renderExpansionSummary(response: ExpansionGraphResponse): void {
    this.renderGraphSummary(response.graph, [
      ["Projection", "Expanded"],
      ["Certified depth", String(response.certified_depth)],
      ["Tried", formatCount(response.attempted_transitions)],
      ["Failed", formatCount(response.failed_transitions)],
    ]);
  }

  private renderGraphSummary(payload: GraphPayload, extras: Array<[string, string]>): void {
    clear(this.graphSummary);
    const values: Array<[string, string]> = [
      ["States", payload.rendered_nodes === payload.total_nodes ? formatCount(payload.total_nodes) : `${formatCount(payload.rendered_nodes)} / ${formatCount(payload.total_nodes)}`],
      ["Transitions", formatCount(payload.total_edges)],
      ["Drawn groups", formatCount(payload.rendered_edge_groups)],
      ...extras,
    ];
    for (const [label, value] of values) {
      const chip = document.createElement("div");
      chip.className = "summary-chip";
      chip.append(textElement("span", label), textElement("strong", value));
      this.graphSummary.append(chip);
    }
    this.graphSummary.append(this.structureDisclosure(payload));
    if (payload.truncated) {
      const truncated = textElement("span", "View capped", "truncated-chip");
      truncated.title = "The graph is larger than the drawing cap; totals still reflect the complete projection.";
      this.graphSummary.append(truncated);
    }
  }

  private structureDisclosure(payload: GraphPayload): HTMLDetailsElement {
    const values: Array<[string, number]> = [
      ["Self-loops", payload.diagnostics.self_loop_groups],
      ["Parallel", payload.diagnostics.parallel_function_groups],
      ["Convergent", payload.diagnostics.convergent_nodes],
      ["Cycles", payload.diagnostics.directed_cycle_groups],
    ];
    const total = values.reduce((sum, [, value]) => sum + value, 0);
    const disclosure = document.createElement("details");
    disclosure.className = "structure-disclosure";
    const summary = document.createElement("summary");
    summary.setAttribute("aria-label", `Show structural diagnostics; ${total} total`);
    summary.append(textElement("span", "Structure"), textElement("strong", formatCount(total)));
    const list = document.createElement("dl");
    list.className = "structure-diagnostics";
    for (const [label, value] of values) appendDefinition(list, label, formatCount(value));
    disclosure.append(summary, list);
    return disclosure;
  }

  private renderInspector(selection: GraphSelection): void {
    clear(this.inspector);
    if (selection.kind === "none") {
      this.inspector.append(textElement("p", "Click a state or transition in the graph.", "muted-copy"));
      this.clearPathButton.disabled = !this.graph.hasGraph;
      return;
    }
    this.clearPathButton.disabled = false;
    if (selection.kind === "edge") {
      const header = document.createElement("div");
      header.className = "inspector-heading";
      header.append(textElement("span", "Transition", "inspector-kind"), textElement("strong", `#${selection.edge.source} → #${selection.edge.target}`));
      const functionList = document.createElement("div");
      functionList.className = "function-chip-list";
      for (const name of selection.edge.function_names) functionList.append(textElement("code", name));
      this.inspector.append(header, textElement("span", "Functions on this edge group", "field-label"), functionList);
      if (selection.edge.function_names.length > 1) {
        this.inspector.append(textElement("p", "These functions converge on the same state transition.", "inspector-note"));
      }
      return;
    }

    const header = document.createElement("div");
    header.className = "inspector-heading";
    const title = document.createElement("div");
    title.append(textElement("span", selection.node.role.replaceAll("_", " "), "inspector-kind"));
    title.append(textElement("strong", `State #${selection.node.id}`));
    header.append(title, textElement("span", `depth ${selection.node.depth}`, "depth-tag"));
    const value = textElement("code", selection.node.value, "inspector-value");
    this.inspector.append(header, value);

    const connectionStats = document.createElement("div");
    connectionStats.className = "connection-stats";
    connectionStats.append(
      textElement("span", `${selection.incoming.length} incoming`),
      textElement("span", `${selection.outgoing.length} outgoing`),
    );
    this.inspector.append(connectionStats);

    const pathHeading = document.createElement("div");
    pathHeading.className = "path-heading";
    pathHeading.append(textElement("span", "Root-to-state path", "field-label"), textElement("strong", `${selection.path.length} step${selection.path.length === 1 ? "" : "s"}`));
    this.inspector.append(pathHeading);
    if (selection.path.length === 0) {
      this.inspector.append(textElement("p", selection.node.role === "root" ? "This state is a graph root." : "No rendered root path reaches this state.", "inspector-note"));
      return;
    }
    const pathList = document.createElement("ol");
    pathList.className = "path-list";
    for (const step of selection.path) {
      const row = document.createElement("li");
      row.append(textElement("code", step.functionName));
      row.append(textElement("span", `#${step.from.id} → #${step.to.id}`));
      if (step.edge.function_names.length > 1) row.append(textElement("small", `+${step.edge.function_names.length - 1} parallel`));
      pathList.append(row);
    }
    this.inspector.append(pathList);
  }

  private renderCandidates(): void {
    clear(this.candidateList);
    this.candidateCount.textContent = formatCount(this.candidates.length);
    const visible = this.candidates.slice(0, this.visibleCandidateCount);
    for (const candidate of visible) {
      const card = document.createElement("div");
      card.className = "candidate-card";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "candidate-inspect";
      const top = document.createElement("span");
      top.className = "candidate-top";
      top.append(
        textElement("span", `d${candidate.distance}`, "distance-badge"),
        textElement("span", candidate.certified ? "Certified" : "Provisional", candidate.certified ? "certified-tag" : "provisional-tag"),
        textElement("span", `#${candidate.node_id}`, "node-tag"),
      );
      button.append(top, textElement("code", candidate.output, "candidate-output"));
      button.append(textElement("span", candidate.function_names.join(" → ") || "identity", "candidate-path"));
      button.addEventListener("click", () => {
        this.graph.focusNode(candidate.node_id);
        this.closeMobileRails();
      });
      const seed = document.createElement("button");
      seed.type = "button";
      seed.className = "candidate-seed";
      seed.textContent = "Use as next root";
      seed.addEventListener("click", () => {
        this.expansionInput.value = candidate.output_serialized;
        const paletteReady = this.selectCompatibleFunctions(10);
        this.renderFunctionPalette();
        this.expansionInput.focus();
        this.expansionControls.scrollIntoView({ behavior: "smooth", block: "start" });
        if (paletteReady) {
          this.showNotice(
            `State #${candidate.node_id} is now the expansion root; its compatible palette is ready.`,
            "info",
          );
        }
      });
      card.append(button, seed);
      this.candidateList.append(card);
    }
    setHidden(this.moreCandidates, visible.length >= this.candidates.length);
    if (this.candidates.length === 0) {
      this.candidateList.append(textElement("p", "No candidate tasks were reached within this budget.", "muted-copy"));
    }
  }

  private setGraphLoading(loading: boolean, title = "Building graph", detail = "Replaying registered functions…"): void {
    this.graphLoadingTitle.textContent = title;
    this.graphLoadingDetail.textContent = detail;
    setHidden(this.graphLoading, !loading);
    requiredElement<HTMLElement>(".graph-stage").classList.toggle("is-loading", loading);
  }

  private resetWorkspaceGraph(): void {
    this.graph.clear();
    this.displayedGraphMode = null;
    this.canvasEmptyTitle.textContent = "Select a task to trace its witness";
    this.canvasEmptyCopy.textContent =
      "Choose a corpus task on the left. Its states and functions will appear here as an interactive directed graph.";
    this.canvasHint.textContent =
      "Drag nodes · scroll to zoom · click a state to trace its shortest path";
    setHidden(this.canvasEmpty, false);
    setHidden(this.graphSummary, true);
    setHidden(this.candidateSection, true);
    this.candidates = [];
    this.fitGraphButton.disabled = true;
    this.resetLayoutButton.disabled = true;
    this.clearPathButton.disabled = true;
    this.setGraphLoading(false);
  }

  private showExpansionPrompt(): void {
    this.graphController?.abort();
    this.graphController = null;
    this.graph.clear();
    this.displayedGraphMode = null;
    this.canvasEmptyTitle.textContent = "Configure a bounded expansion";
    this.canvasEmptyCopy.textContent =
      "Choose a root, compatible function palette, and budget in Graph tools, then run the expansion.";
    this.canvasHint.textContent = "Expand mode · configure and run from Graph tools";
    setHidden(this.canvasEmpty, false);
    setHidden(this.graphSummary, true);
    this.fitGraphButton.disabled = true;
    this.resetLayoutButton.disabled = true;
    this.clearPathButton.disabled = true;
    this.setGraphLoading(false);
  }

  private showNotice(message: string, kind: NoticeKind, autoDismiss = true): void {
    const notice = document.createElement("div");
    notice.className = `notice ${kind}`;
    const copy = document.createElement("div");
    copy.append(textElement("strong", kind === "error" ? "Something went wrong" : kind === "warning" ? "Heads up" : "Note"));
    copy.append(textElement("span", message));
    const closeButton = textElement("button", "×", "notice-close");
    closeButton.type = "button";
    closeButton.setAttribute("aria-label", "Dismiss notification");
    closeButton.addEventListener("click", () => notice.remove());
    notice.append(copy, closeButton);
    this.noticeStack.append(notice);
    if (autoDismiss) window.setTimeout(() => notice.remove(), kind === "error" ? 8_000 : 5_500);
  }

  private setHeader(state: "loading" | "ready" | "warning" | "error", message: string): void {
    this.headerContext.textContent = message;
    this.headerStatusDot.className = `status-dot ${state}`;
  }

  private renderNoSources(): void {
    this.setHeader("warning", "No corpus sources discovered");
    clear(this.sourceCard);
    this.sourceCard.append(textElement("strong", "No corpora found"), textElement("p", "Add a manifest-backed corpus or JSONL gzip under a configured corpus path."));
    this.sourceSelect.disabled = true;
    setHidden(this.sourceAction, true);
    this.renderTaskList();
  }

  private renderFatalError(message: string): void {
    clear(this.taskList);
    const error = document.createElement("div");
    error.className = "list-empty error-state";
    error.append(textElement("strong", "The app could not start"), textElement("p", message));
    this.taskList.append(error);
    this.showNotice(message, "error", false);
  }

  private abortRequests(): void {
    this.taskController?.abort();
    this.detailController?.abort();
    this.graphController?.abort();
    this.taskController = null;
    this.detailController = null;
    this.graphController = null;
  }

  private readLastSource(): string | null {
    try {
      return window.localStorage.getItem(LAST_SOURCE_KEY);
    } catch {
      return null;
    }
  }

  private writeLastSource(sourceId: string): void {
    try {
      window.localStorage.setItem(LAST_SOURCE_KEY, sourceId);
    } catch {
      // Persistence is an enhancement; restricted browser storage is harmless.
    }
  }

  private toolsAreDrawer(): boolean {
    return window.matchMedia("(max-width: 980px)").matches;
  }

  private openMobileRail(rail: "browser" | "tools"): void {
    document.body.classList.toggle("show-browser", rail === "browser");
    document.body.classList.toggle("show-tools", rail === "tools");
    this.toggleBrowserButton.setAttribute("aria-expanded", String(rail === "browser"));
    this.toggleDetailsButton.setAttribute("aria-expanded", String(rail === "tools"));
  }

  private closeMobileRails(): void {
    document.body.classList.remove("show-browser", "show-tools");
    this.toggleBrowserButton.setAttribute("aria-expanded", "false");
    this.toggleDetailsButton.setAttribute("aria-expanded", "false");
  }
}

const app = new TrajectoryGraphApp();
void app.start();

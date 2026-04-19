(() => {
    let sessionId = null;
    let isStreaming = false;
    let interruptPending = false;
    let activeStreamPromise = null;
    let activeAssistantContainer = null;
    let sessions = [];
    let userScrolledUp = false;

    const $app = document.getElementById("app");
    const $homeView = document.getElementById("home-view");
    const $workspaceView = document.getElementById("workspace-view");
    const $chatView = document.getElementById("chat-view");
    const $messages = document.getElementById("messages");
    const $input = document.getElementById("chat-input");
    const $sendBtn = document.getElementById("send-btn");
    const $interveneBtn = document.getElementById("intervene-btn");
    const $homeBtn = document.getElementById("home-btn");
    const $newSessionBtn = document.getElementById("new-session-btn");
    const $sidebarNewSessionBtn = document.getElementById("sidebar-new-session-btn");
    const $sessionList = document.getElementById("session-list");
    const $statusText = document.getElementById("status-text");
    const $composerState = document.getElementById("composer-state");
    const $composerHint = document.getElementById("composer-hint");
    const $welcomeTemplate = document.getElementById("welcome-template");
    const $logoHomeBtn = document.getElementById("logo-home-btn");
    const $sidebarToggle = document.getElementById("sidebar-toggle");
    const $sessionSidebar = document.querySelector(".session-sidebar");
    const $sharedContextSidebar = document.getElementById("shared-context-sidebar");
    const $sharedContextList = document.getElementById("shared-context-list");
    const $sharedContextCount = document.getElementById("shared-context-count");
    const $sharedContextToggle = document.getElementById("shared-context-toggle");
    const $activeAgentLabel = document.getElementById("active-agent-label");
    const $activeRunBanner = document.getElementById("active-run-banner");
    const $activeRunBannerStatus = document.getElementById("active-run-banner-status");
    const $activeRunBannerGoal = document.getElementById("active-run-banner-goal");
    const $activeRunBannerMeta = document.getElementById("active-run-banner-meta");
    const $headerOpenRunsBtn = document.getElementById("header-open-runs-btn");
    const $sidebarTabButtons = document.querySelectorAll(".sidebar-tab-btn");
    const $runsPanel = document.getElementById("runs-panel");
    const $contextPanel = document.getElementById("context-panel");
    const $learningPanel = document.getElementById("learning-panel");
    const $memoryView = document.getElementById("view-memory");
    const $primaryAttentionPanel = document.getElementById("primary-attention-panel");
    const $approvalInboxShell = document.getElementById("approval-inbox-shell");
    const $runSessionSummary = document.getElementById("run-session-summary");
    const $approvalInboxList = document.getElementById("approval-inbox-list");
    const $approvalInboxCount = document.getElementById("approval-inbox-count");
    const $runActiveCard = document.getElementById("run-active-card");
    const $runHistoryList = document.getElementById("run-history-list");
    const $runHistoryCount = document.getElementById("run-history-count");
    const $runDetailPanel = document.getElementById("run-detail-panel");
    const $runDetailPill = document.getElementById("run-detail-pill");
    const $runPanelRefreshBtn = document.getElementById("run-panel-refresh-btn");
    const $uploadTriggerBtn = document.getElementById("upload-trigger-btn");
    const $uploadInput = document.getElementById("upload-input");
    const $composerAttachments = document.getElementById("composer-attachments");
    const $sessionUploadsShell = document.getElementById("session-uploads-shell");
    const $sessionUploadsList = document.getElementById("session-uploads-list");
    const $sessionUploadsCount = document.getElementById("session-uploads-count");
    const $sessionUploadsRefreshBtn = document.getElementById("session-uploads-refresh-btn");
    const $uploadDropzone = document.getElementById("upload-dropzone");
    const $filePreviewModal = document.getElementById("file-preview-modal");
    const $filePreviewTitle = document.getElementById("file-preview-title");
    const $filePreviewMeta = document.getElementById("file-preview-meta");
    const $filePreviewBody = document.getElementById("file-preview-body");
    const $filePreviewCloseBtn = document.getElementById("file-preview-close-btn");
    const $filePreviewOpenBtn = document.getElementById("file-preview-open-btn");
    const $filePreviewDownloadBtn = document.getElementById("file-preview-download-btn");

    let sharedContextEntries = [];
    let sharedContextRefreshTimer = null;
    let markdownConfigured = false;
    let markdownUnavailableWarned = false;
    const DEFAULT_AGENT_ID = "system-default-agent";
    const DEFAULT_AGENT_LABEL = "SYSTEM_AGENT";
    let currentAgentId = DEFAULT_AGENT_ID;
    let agentCatalog = new Map([[DEFAULT_AGENT_ID, DEFAULT_AGENT_LABEL]]);
    let marketplaceAgentCatalog = new Map();
    let defaultAgentToolset = [];
    let agentGridPollTimer = null;
    let agentModalPollTimer = null;
    let sessionRuns = [];
    let selectedRunId = null;
    let activeRunId = null;
    let selectedRunSteps = [];
    let selectedRunTimeline = [];
    let selectedRunArtifacts = [];
    let selectedRunApprovals = [];
    let sessionPendingApprovals = [];
    let selectedRunTree = null;
    let runRefreshTimer = null;
    let approvalActionIds = new Set();
    let activeSidebarTab = "runs";
    let sessionUploads = [];
    let composerAttachments = [];
    let uploadDragDepth = 0;
    let activeFilePreviewRequestId = 0;
    const assistantArtifactShelves = new WeakMap();
    let conversationAssistantTurns = [];
    let conversationArtifactsByRunId = new Map();
    let conversationArtifactFetchKeys = new Map();
    let userProfileSnapshot = null;
    let userMemoryEntries = [];
    let learnedWorkflowCandidates = [];
    let memoryViewError = "";
    let memoryViewLoading = false;
    let memoryQuery = "";
    let memoryTypeFilter = "";
    let memoryIncludeSuperseded = false;
    // 当前登录账号信息（bootstrap 后赋值）
    let currentAccount = null;
    const agentModalState = {
        agentId: "",
        isSystem: false,
        searchResults: [],
        selectedPackages: new Set(),
        installedSkills: [],
        skillInstallStatus: null,
        activeSkillGroupIndex: 0,
    };
    const RUN_ACTIVE_STATUSES = new Set(["queued", "running", "waiting_approval"]);
    let featureFlags = {
        enable_durable_runs: true,
        enable_run_trace: true,
        enable_approval_flow: true,
    };
    const STATIC_ASSET_VERSION = String(window.__CLAVI_STATIC_VERSION__ || "20260419-integrations-controls-3");

    function withAssetVersion(url) {
        const separator = url.includes("?") ? "&" : "?";
        return `${url}${separator}v=${encodeURIComponent(STATIC_ASSET_VERSION)}`;
    }

    function isFeatureEnabled(flagName) {
        return featureFlags?.[flagName] !== false;
    }

    async function refreshFeatureFlags() {
        try {
            const flags = await fetchJson("/api/features");
            if (flags && typeof flags === "object") {
                featureFlags = {
                    ...featureFlags,
                    ...flags,
                };
            }
        } catch (error) {
            console.warn("Failed to load feature flags. Falling back to all features enabled.", error);
        }
        applyFeatureFlags();
    }

    function applyFeatureFlags() {
        const durableRunsEnabled = isFeatureEnabled("enable_durable_runs");
        $sidebarTabButtons.forEach((button) => {
            if (button.dataset.sidebarTab === "runs") {
                button.classList.toggle("hidden", !durableRunsEnabled);
            }
        });
        if (!durableRunsEnabled) {
            clearRunRefreshTimer();
            clearRunState();
            if ($headerOpenRunsBtn) {
                $headerOpenRunsBtn.classList.add("hidden");
                $headerOpenRunsBtn.classList.remove("inline-flex");
            }
        }
        setSidebarTab(durableRunsEnabled ? activeSidebarTab : "context");
    }

    function createWelcome() {
        return document.createElement("div"); // Deprecated
    }

    function showHomeView() {
        // Obsolete function, design uses permanent workspace view now
    }

    function showChatView() {
        // Obsolete function, design uses permanent workspace view now
    }

    function renderHome() {
        // Obsolete function, default directly to workspace
    }

    function autoResizeInput() {
        $input.style.height = "auto";
        $input.style.height = `${Math.min($input.scrollHeight, 168)}px`;
    }

    function setStatus(text) {
        $statusText.textContent = text;
    }

    function setComposerState(text) {
        $composerState.textContent = text;
    }

    function setComposerHint(text) {
        if ($composerHint) {
            $composerHint.textContent = text;
        }
    }

    function setComposerPlaceholder(text) {
        if ($input) {
            $input.placeholder = text;
        }
    }

    function getComposerCopy() {
        const rootRuns = getRootRuns();
        const activeRun = getCurrentRootRun();
        const waitingOnUser =
            sessionPendingApprovals.length > 0 ||
            activeRun?.status === "waiting_approval" ||
            activeRun?.status === "interrupted";

        if (!sessionId || !rootRuns.length) {
            return {
                label: "START",
                icon: "send",
                title: "Start task (Enter)",
                placeholder: "Describe the task you want the agent to handle... (Shift+Enter for newline)",
                hint: "Press Enter to start a task. Shift+Enter for newline.",
            };
        }

        if (waitingOnUser) {
            return {
                label: "REPLY",
                icon: "send",
                title: "Reply (Enter)",
                placeholder: "Answer the question, add clarification, or tell the agent how to continue...",
                hint: "The task is waiting on your input. You can reply here or act on the request above.",
            };
        }

        if (activeRun) {
            return {
                label: "CONTINUE",
                icon: "send",
                title: "Continue task (Enter)",
                placeholder: "Add more guidance for the current task...",
                hint: "The current task is active. Add guidance only if you want to steer it.",
            };
        }

        return {
            label: "SEND",
            icon: "send",
            title: "Send (Enter)",
            placeholder: "Ask for a follow-up, a revision, or start a new task...",
            hint: "Press Enter to continue this session. Shift+Enter for newline.",
        };
    }

    function updateComposerActionButton() {
        if (!$sendBtn) return;
        const hasTextInput = Boolean($input?.value.trim());
        const readyAttachmentCount = getReadyComposerAttachments().length;
        const hasInput = hasTextInput || readyAttachmentCount > 0;
        const hasBlockingAttachments = hasBlockingComposerAttachments();

        if (isStreaming) {
            $sendBtn.disabled = interruptPending;
            $sendBtn.title = interruptPending ? "Interrupt requested" : "Stop current run";
            $sendBtn.classList.remove("bg-primary", "hover:bg-primary-container", "text-on-primary");
            $sendBtn.classList.add("bg-error", "hover:bg-error/80", "text-on-error");
            $sendBtn.innerHTML = `
                <span>${interruptPending ? "STOPPING" : "STOP"}</span>
                <span class="material-symbols-outlined text-sm">${interruptPending ? "hourglass_top" : "stop"}</span>
            `;
            if ($interveneBtn) {
                $interveneBtn.classList.remove("hidden");
                $interveneBtn.classList.add("inline-flex");
                $interveneBtn.disabled = interruptPending || !hasTextInput;
                $interveneBtn.title = hasTextInput
                    ? "Interrupt the current run and continue with your guidance"
                    : "Type guidance first, then intervene";
            }
            setComposerHint(
                hasTextInput
                    ? "Human in loop ready. Intervene will stop the current run and continue with your guidance."
                    : "Agent is running. Type corrective guidance, then press Intervene."
            );
            setComposerPlaceholder("Type guidance here if you want to redirect the current task...");
            updateUploadControlsState();
            return;
        }

        const composerCopy = getComposerCopy();
        $sendBtn.disabled = !hasInput || hasBlockingAttachments;
        if (hasBlockingAttachments) {
            $sendBtn.title = "请先等待上传完成，或移除失败的附件";
        } else if (!hasInput) {
            $sendBtn.title = "先输入消息或附加文件";
        } else {
            $sendBtn.title = composerCopy.title;
        }
        $sendBtn.classList.remove("bg-error", "hover:bg-error/80", "text-on-error");
        $sendBtn.classList.add("bg-primary", "hover:bg-primary-container", "text-on-primary");
        $sendBtn.innerHTML = `
            <span>${composerCopy.label}</span>
            <span class="material-symbols-outlined text-sm">${composerCopy.icon}</span>
        `;
        if ($interveneBtn) {
            $interveneBtn.classList.add("hidden");
            $interveneBtn.classList.remove("inline-flex");
            $interveneBtn.disabled = true;
        }
        setComposerHint(buildComposerAttachmentHint(composerCopy.hint));
        setComposerPlaceholder(composerCopy.placeholder);
        updateUploadControlsState();
    }

    function setStreaming(value) {
        isStreaming = value;
        if (!value) {
            interruptPending = false;
            activeAssistantContainer = null;
        }
        setStatus(value ? "Clavi Agent is responding" : "Session ready");
        setComposerState(value ? "Streaming" : "Ready");
        updateComposerActionButton();
        renderRunBanner();
        scheduleRunRefresh();
    }

    function normalizeAgentId(agentId) {
        return agentId || DEFAULT_AGENT_ID;
    }

    function applyAgentCatalog(agents) {
        if (!Array.isArray(agents)) return;
        agentCatalog = new Map([[DEFAULT_AGENT_ID, DEFAULT_AGENT_LABEL]]);
        marketplaceAgentCatalog = new Map();
        agents.forEach((agent) => {
            if (!agent || !agent.id) return;
            agentCatalog.set(agent.id, (agent.name || agent.id).trim());
            marketplaceAgentCatalog.set(agent.id, agent);
        });
        const defaultTemplate = marketplaceAgentCatalog.get(DEFAULT_AGENT_ID);
        defaultAgentToolset = Array.isArray(defaultTemplate?.tools) ? defaultTemplate.tools : [];
        if ($activeAgentLabel) {
            $activeAgentLabel.textContent = getAgentDisplayName(currentAgentId);
        }
        if (sessions.length) {
            renderSessionList();
        }
    }

    async function refreshAgentCatalog() {
        try {
            const agents = await fetchJson("/api/agents");
            applyAgentCatalog(agents);
        } catch (error) {
            console.warn("Failed to refresh agent catalog:", error);
        }
    }

    function getAgentDisplayName(agentId = currentAgentId) {
        const normalizedId = normalizeAgentId(agentId);
        return agentCatalog.get(normalizedId) || normalizedId;
    }

    function setActiveAgent(agentId) {
        currentAgentId = normalizeAgentId(agentId);
        if ($activeAgentLabel) {
            $activeAgentLabel.textContent = getAgentDisplayName(currentAgentId);
        }
    }

    function resolveNewSessionAgentId() {
        if (!sessionId) {
            return DEFAULT_AGENT_ID;
        }
        return normalizeAgentId(currentAgentId);
    }

    async function createSessionForSelectedAgent({ openChat = true } = {}) {
        return createSession({
            openChat,
            agent_id: resolveNewSessionAgentId(),
        });
    }

    function truncate(text, limit = 900) {
        return text.length > limit ? `${text.slice(0, limit)}...` : text;
    }

    function formatToolData(data) {
        try {
            return truncate(JSON.stringify(data, null, 2));
        } catch {
            return String(data);
        }
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    function escapeAndFormat(text) {
        let html = escapeHtml(text || "");
        html = html.replace(/```(\w+)?\n?([\s\S]*?)```/g, (_, __, code) => `<pre><code>${code.trim()}</code></pre>`);
        html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
        html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
        html = html.replace(/\n/g, "<br>");
        return html;
    }

    function renderPlainText(text) {
        return escapeHtml(text || "").replace(/\n/g, "<br>");
    }

    function isMarkdownRuntimeReady() {
        const hasMarked =
            typeof window.marked !== "undefined" &&
            window.marked &&
            typeof window.marked.parse === "function";
        const hasPurify =
            typeof window.DOMPurify !== "undefined" &&
            window.DOMPurify &&
            typeof window.DOMPurify.sanitize === "function";

        if (!hasMarked || !hasPurify) {
            if (!markdownUnavailableWarned) {
                markdownUnavailableWarned = true;
                console.warn("Markdown runtime unavailable; fallback rendering is enabled.");
            }
            return false;
        }

        if (!markdownConfigured) {
            if (typeof window.marked.use === "function") {
                window.marked.use({ gfm: true, breaks: true });
            }
            markdownConfigured = true;
        }

        return true;
    }

    function renderMarkdownSafe(text) {
        const raw = text || "";
        if (!raw) return "";

        if (!isMarkdownRuntimeReady()) {
            return escapeAndFormat(raw);
        }

        try {
            const parsed = window.marked.parse(raw);
            return window.DOMPurify.sanitize(parsed);
        } catch (error) {
            console.error("Markdown rendering failed:", error);
            return escapeAndFormat(raw);
        }
    }

    function normalizeMessageBlocks(content) {
        if (!Array.isArray(content)) return [];
        return content.filter((block) => block && typeof block === "object" && typeof block.type === "string");
    }

    function formatFileSize(sizeBytes) {
        if (!Number.isFinite(sizeBytes) || sizeBytes < 0) {
            return "";
        }
        if (sizeBytes < 1024) {
            return `${sizeBytes} B`;
        }
        if (sizeBytes < 1024 * 1024) {
            return `${(sizeBytes / 1024).toFixed(1)} KB`;
        }
        return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
    }

    function generateClientId(prefix = "upload") {
        if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
            return `${prefix}-${crypto.randomUUID()}`;
        }
        return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
    }

    function normalizeUploadRecord(upload) {
        return upload && typeof upload === "object" && typeof upload.id === "string" ? upload : null;
    }

    function guessPreviewKindFromFileMeta(filename, mimeType) {
        const normalizedMime = String(mimeType || "").trim().toLowerCase();
        const normalizedName = String(filename || "").trim().toLowerCase();
        const extension = normalizedName.includes(".") ? normalizedName.split(".").pop() : "";

        if (extension === "md" || extension === "markdown" || normalizedMime === "text/markdown") {
            return "markdown";
        }
        if (normalizedMime.startsWith("text/") || ["txt", "csv", "log"].includes(extension)) {
            return "text";
        }
        if (extension === "json" || normalizedMime === "application/json") {
            return "json";
        }
        if (extension === "html" || extension === "htm" || normalizedMime === "text/html") {
            return "html";
        }
        if (
            normalizedMime.startsWith("image/") ||
            ["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(extension)
        ) {
            return "image";
        }
        if (extension === "pdf" || normalizedMime === "application/pdf") {
            return "pdf";
        }
        if (["doc", "docx", "ppt", "pptx", "xls", "xlsx"].includes(extension)) {
            return "office";
        }
        return "none";
    }

    function formatPreviewKindLabel(previewKind) {
        const labels = {
            markdown: "Markdown",
            text: "文本",
            json: "JSON",
            html: "HTML",
            image: "图片",
            pdf: "PDF",
            office: "Office",
            none: "文件",
        };
        return labels[String(previewKind || "").trim().toLowerCase()] || "文件";
    }

    function buildFileResourceUrls(targetKind, targetId) {
        if (!targetKind || !targetId) {
            return null;
        }
        const basePath = targetKind === "upload"
            ? `/api/uploads/${encodeURIComponent(targetId)}`
            : `/api/artifacts/${encodeURIComponent(targetId)}`;
        return {
            previewUrl: `${basePath}/preview`,
            openUrl: `${basePath}?disposition=inline`,
            downloadUrl: basePath,
        };
    }

    function createFileResourceDescriptor({ targetKind, targetId, displayName, previewKind = "none" }) {
        if (!targetKind || !targetId) {
            return null;
        }
        const urls = buildFileResourceUrls(targetKind, targetId);
        if (!urls) {
            return null;
        }
        return {
            targetKind,
            targetId,
            displayName: displayName || "未命名文件",
            previewKind: previewKind || "none",
            ...urls,
        };
    }

    function createUploadFileDescriptor(upload) {
        if (!upload?.id) {
            return null;
        }
        return createFileResourceDescriptor({
            targetKind: "upload",
            targetId: upload.id,
            displayName: upload.original_name || upload.safe_name || upload.id,
            previewKind: guessPreviewKindFromFileMeta(
                upload.original_name || upload.safe_name || upload.id,
                upload.mime_type
            ),
        });
    }

    function createArtifactFileDescriptor(artifact) {
        if (!artifact?.id) {
            return null;
        }
        if (!["workspace_file", "document"].includes(String(artifact.artifact_type || "").trim())) {
            return null;
        }
        return createFileResourceDescriptor({
            targetKind: "artifact",
            targetId: artifact.id,
            displayName: artifact.display_name || artifact.uri || artifact.id,
            previewKind: artifact.preview_kind || guessPreviewKindFromFileMeta(
                artifact.display_name || artifact.uri || artifact.id,
                artifact.mime_type
            ),
        });
    }

    function encodePreviewName(value) {
        try {
            return encodeURIComponent(String(value || ""));
        } catch {
            return "";
        }
    }

    function decodePreviewName(value) {
        try {
            return decodeURIComponent(String(value || ""));
        } catch {
            return String(value || "");
        }
    }

    function renderFileActionButtons(fileDescriptor, { buttonClassName = "" } = {}) {
        if (!fileDescriptor) {
            return "";
        }
        const actionClass =
            buttonClassName ||
            "border-outline-variant/20 bg-surface-container-highest text-on-surface hover:border-primary/40 hover:text-primary";

        return `
            <div class="flex flex-wrap items-center gap-2">
                <button
                    type="button"
                    data-file-preview-target="${escapeHtml(fileDescriptor.targetKind)}"
                    data-file-preview-id="${escapeHtml(fileDescriptor.targetId)}"
                    data-file-preview-name="${encodePreviewName(fileDescriptor.displayName)}"
                    class="inline-flex items-center gap-2 rounded-xl border px-3 py-1.5 text-[11px] font-semibold transition-colors ${actionClass}"
                >
                    <span class="material-symbols-outlined text-sm">preview</span>
                    <span>预览</span>
                </button>
                <a
                    href="${fileDescriptor.openUrl}"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="inline-flex items-center gap-2 rounded-xl border px-3 py-1.5 text-[11px] font-semibold transition-colors ${actionClass}"
                >
                    <span class="material-symbols-outlined text-sm">open_in_new</span>
                    <span>打开</span>
                </a>
                <a
                    href="${fileDescriptor.downloadUrl}"
                    class="inline-flex items-center gap-2 rounded-xl border px-3 py-1.5 text-[11px] font-semibold transition-colors ${actionClass}"
                >
                    <span class="material-symbols-outlined text-sm">download</span>
                    <span>下载</span>
                </a>
            </div>
        `;
    }

    function closeFilePreview() {
        activeFilePreviewRequestId += 1;
        if (!$filePreviewModal) {
            return;
        }
        $filePreviewModal.classList.add("hidden");
        $filePreviewModal.classList.remove("flex");
        if ($filePreviewTitle) {
            $filePreviewTitle.textContent = "文件预览";
        }
        if ($filePreviewMeta) {
            $filePreviewMeta.textContent = "";
        }
        if ($filePreviewBody) {
            $filePreviewBody.innerHTML = "";
        }
        if ($filePreviewOpenBtn) {
            $filePreviewOpenBtn.classList.add("hidden");
            $filePreviewOpenBtn.removeAttribute("href");
        }
        if ($filePreviewDownloadBtn) {
            $filePreviewDownloadBtn.classList.add("hidden");
            $filePreviewDownloadBtn.removeAttribute("href");
        }
    }

    function renderFilePreview(preview, fallbackName = "") {
        if (!$filePreviewBody) {
            return;
        }

        const displayName = preview?.display_name || fallbackName || "文件预览";
        const previewKind = String(preview?.preview_kind || "none").trim().toLowerCase();
        const metaParts = [];
        const previewKindLabel = formatPreviewKindLabel(previewKind);
        const sizeLabel = formatFileSize(Number(preview?.size_bytes));
        if (previewKindLabel) {
            metaParts.push(previewKindLabel);
        }
        if (preview?.mime_type) {
            metaParts.push(preview.mime_type);
        }
        if (sizeLabel) {
            metaParts.push(sizeLabel);
        }

        if ($filePreviewTitle) {
            $filePreviewTitle.textContent = displayName;
        }
        if ($filePreviewMeta) {
            $filePreviewMeta.textContent = metaParts.join(" · ");
        }
        if ($filePreviewOpenBtn) {
            if (preview?.open_url) {
                $filePreviewOpenBtn.href = preview.open_url;
                $filePreviewOpenBtn.classList.remove("hidden");
            } else {
                $filePreviewOpenBtn.classList.add("hidden");
                $filePreviewOpenBtn.removeAttribute("href");
            }
        }
        if ($filePreviewDownloadBtn) {
            if (preview?.download_url) {
                $filePreviewDownloadBtn.href = preview.download_url;
                $filePreviewDownloadBtn.classList.remove("hidden");
            } else {
                $filePreviewDownloadBtn.classList.add("hidden");
                $filePreviewDownloadBtn.removeAttribute("href");
            }
        }

        const notices = [];
        if (preview?.truncated) {
            notices.push("预览内容较长，当前仅展示前一部分。");
        }
        if (preview?.note) {
            notices.push(preview.note);
        }

        let previewContent = `
            <div class="rounded-2xl border border-dashed border-outline-variant/20 bg-surface-container-high/50 px-4 py-5 text-sm text-on-surface-variant">
                当前文件暂无可用预览，请直接打开原文件或下载。
            </div>
        `;

        if (typeof preview?.text_content === "string" && ["markdown", "text", "json"].includes(previewKind)) {
            previewContent = previewKind === "markdown"
                ? `
                    <div class="rounded-2xl border border-outline-variant/15 bg-surface-container-high/60 px-5 py-5">
                        <div class="prose prose-invert max-w-none text-sm leading-7">
                            ${renderMarkdownSafe(preview.text_content)}
                        </div>
                    </div>
                `
                : `
                    <pre class="overflow-x-auto whitespace-pre-wrap rounded-2xl border border-outline-variant/15 bg-black/30 px-4 py-4 text-[12px] leading-6 text-on-surface">${escapeHtml(preview.text_content)}</pre>
                `;
        } else if (preview?.preview_supported && previewKind === "image" && preview?.inline_url) {
            previewContent = `
                <div class="overflow-auto rounded-2xl border border-outline-variant/15 bg-black/20 px-3 py-3">
                    <img src="${preview.inline_url}" alt="${escapeHtml(displayName)}" class="mx-auto max-h-[65vh] w-auto rounded-xl object-contain" />
                </div>
            `;
        } else if (preview?.preview_supported && previewKind === "pdf" && preview?.inline_url) {
            previewContent = `
                <div class="overflow-hidden rounded-2xl border border-outline-variant/15 bg-black/20">
                    <iframe src="${preview.inline_url}" title="${escapeHtml(displayName)}" class="h-[70vh] w-full bg-white"></iframe>
                </div>
            `;
        }

        const noticesHtml = notices.length
            ? notices
                .map(
                    (notice) => `
                        <div class="rounded-2xl border border-primary/15 bg-primary/10 px-4 py-3 text-[12px] leading-6 text-on-surface">
                            ${escapeHtml(notice)}
                        </div>
                    `
                )
                .join("")
            : "";

        $filePreviewBody.innerHTML = `
            <div class="space-y-3">
                ${noticesHtml}
                ${previewContent}
            </div>
        `;
    }

    async function openFilePreview(targetKind, targetId, fallbackName = "") {
        if (!$filePreviewModal || !$filePreviewBody) {
            return;
        }
        const urls = buildFileResourceUrls(targetKind, targetId);
        if (!urls) {
            return;
        }

        const requestId = ++activeFilePreviewRequestId;
        $filePreviewModal.classList.remove("hidden");
        $filePreviewModal.classList.add("flex");
        if ($filePreviewTitle) {
            $filePreviewTitle.textContent = fallbackName || "文件预览";
        }
        if ($filePreviewMeta) {
            $filePreviewMeta.textContent = "正在加载预览…";
        }
        if ($filePreviewOpenBtn) {
            $filePreviewOpenBtn.href = urls.openUrl;
            $filePreviewOpenBtn.classList.remove("hidden");
        }
        if ($filePreviewDownloadBtn) {
            $filePreviewDownloadBtn.href = urls.downloadUrl;
            $filePreviewDownloadBtn.classList.remove("hidden");
        }
        $filePreviewBody.innerHTML = `
            <div class="rounded-2xl border border-outline-variant/15 bg-surface-container-high/60 px-4 py-6 text-sm text-on-surface-variant">
                正在加载文件预览…
            </div>
        `;

        try {
            const preview = await fetchJson(urls.previewUrl);
            if (requestId !== activeFilePreviewRequestId) {
                return;
            }
            renderFilePreview(preview, fallbackName);
        } catch (error) {
            if (requestId !== activeFilePreviewRequestId) {
                return;
            }
            if ($filePreviewMeta) {
                $filePreviewMeta.textContent = "预览加载失败";
            }
            $filePreviewBody.innerHTML = `
                <div class="rounded-2xl border border-error/20 bg-error/10 px-4 py-5 text-sm text-on-surface">
                    无法加载文件预览：${escapeHtml(error.message || "未知错误")}
                </div>
            `;
        }
    }

    function openFilePreviewFromDataset(dataset) {
        const targetKind = dataset?.filePreviewTarget;
        const targetId = dataset?.filePreviewId;
        if (!targetKind || !targetId) {
            return;
        }
        openFilePreview(targetKind, targetId, decodePreviewName(dataset.filePreviewName || ""));
    }

    function createUploadedFileBlock(upload) {
        return {
            type: "uploaded_file",
            upload_id: upload.id,
            original_name: upload.original_name || "",
            safe_name: upload.safe_name || "",
            relative_path: upload.relative_path || "",
            mime_type: upload.mime_type || "",
            size_bytes: Number.isFinite(Number(upload.size_bytes)) ? Number(upload.size_bytes) : null,
            checksum: upload.checksum || "",
        };
    }

    function getReadyComposerAttachments() {
        return composerAttachments.filter((attachment) => attachment.status === "ready" && attachment.upload?.id);
    }

    function hasBlockingComposerAttachments() {
        return composerAttachments.some(
            (attachment) => attachment.status === "uploading" || attachment.status === "error"
        );
    }

    function formatUploadProgress(progress) {
        const normalized = Number.isFinite(progress) ? Math.max(0, Math.min(100, Math.round(progress))) : 0;
        return `${normalized}%`;
    }

    function getComposerAttachmentStatusText(attachment) {
        if (attachment.status === "ready") {
            return attachment.source === "session" ? "已附加" : "上传完成";
        }
        if (attachment.status === "error") {
            return attachment.errorMessage || "上传失败";
        }
        return `上传中 ${formatUploadProgress(attachment.progress)}`;
    }

    function buildComposerAttachmentHint(defaultHint) {
        const uploadingCount = composerAttachments.filter((attachment) => attachment.status === "uploading").length;
        if (uploadingCount > 0) {
            return `正在上传 ${uploadingCount} 个文件，上传完成后才能发送。`;
        }

        const failedCount = composerAttachments.filter((attachment) => attachment.status === "error").length;
        if (failedCount > 0) {
            return `有 ${failedCount} 个附件上传失败，请移除后重新上传。`;
        }

        const readyCount = getReadyComposerAttachments().length;
        if (readyCount > 0) {
            return `下一条消息将附带 ${readyCount} 个文件。`;
        }

        return defaultHint;
    }

    function updateUploadControlsState() {
        if ($uploadTriggerBtn) {
            $uploadTriggerBtn.disabled = isStreaming;
            $uploadTriggerBtn.title = isStreaming ? "任务执行中，暂不支持新增上传" : "上传文件";
        }
        if ($uploadInput) {
            $uploadInput.disabled = isStreaming;
        }
        if ($sessionUploadsRefreshBtn) {
            $sessionUploadsRefreshBtn.disabled = !sessionId;
            $sessionUploadsRefreshBtn.title = sessionId ? "同步文件列表" : "先创建或打开会话";
        }
    }

    function setUploadDropzoneActive(active) {
        if (!$uploadDropzone) {
            return;
        }
        $uploadDropzone.classList.toggle("hidden", !active);
        $uploadDropzone.classList.toggle("flex", active);
    }

    function sortUploadsByCreatedAt(uploads) {
        return [...uploads].sort((left, right) => {
            const leftTime = new Date(left?.created_at || 0).getTime();
            const rightTime = new Date(right?.created_at || 0).getTime();
            const normalizedLeftTime = Number.isNaN(leftTime) ? 0 : leftTime;
            const normalizedRightTime = Number.isNaN(rightTime) ? 0 : rightTime;
            if (normalizedRightTime !== normalizedLeftTime) {
                return normalizedRightTime - normalizedLeftTime;
            }
            return String(right?.id || "").localeCompare(String(left?.id || ""));
        });
    }

    function upsertSessionUploads(uploads) {
        const uploadMap = new Map(sessionUploads.map((upload) => [upload.id, upload]));
        uploads.forEach((upload) => {
            const normalized = normalizeUploadRecord(upload);
            if (normalized) {
                uploadMap.set(normalized.id, normalized);
            }
        });
        sessionUploads = sortUploadsByCreatedAt([...uploadMap.values()]);
        renderSessionUploadsPanel();
    }

    function clearSessionUploadsPanel() {
        sessionUploads = [];
        renderSessionUploadsPanel();
    }

    function renderSessionUploadsPanel() {
        if (!$sessionUploadsList) {
            return;
        }

        if ($sessionUploadsCount) {
            $sessionUploadsCount.textContent = String(sessionUploads.length);
        }
        if ($sessionUploadsShell) {
            $sessionUploadsShell.classList.toggle("opacity-80", !sessionId);
        }

        if (!sessionId) {
            $sessionUploadsList.innerHTML = `
                <div class="rounded-2xl border border-dashed border-outline-variant/20 bg-surface-container-high/60 px-4 py-5 text-sm text-on-surface-variant">
                    先创建或打开一个会话，再把文件上传进来。
                </div>
            `;
            updateUploadControlsState();
            return;
        }

        if (!sessionUploads.length) {
            $sessionUploadsList.innerHTML = `
                <div class="rounded-2xl border border-dashed border-outline-variant/20 bg-surface-container-high/60 px-4 py-5 text-sm text-on-surface-variant">
                    当前会话还没有上传文件。你可以点击输入框下方的“上传文件”，也可以直接把文件拖进对话区。
                </div>
            `;
            updateUploadControlsState();
            return;
        }

        const attachedUploadIds = new Set(
            getReadyComposerAttachments()
                .map((attachment) => attachment.upload?.id || "")
                .filter(Boolean)
        );

        $sessionUploadsList.innerHTML = sessionUploads
            .map((upload) => {
                const uploadDescriptor = createUploadFileDescriptor(upload);
                const displayName = escapeHtml(upload.original_name || upload.safe_name || upload.id || "未命名文件");
                const metaParts = [];
                if (upload.mime_type) {
                    metaParts.push(escapeHtml(upload.mime_type));
                }
                const sizeLabel = formatFileSize(Number(upload.size_bytes));
                if (sizeLabel) {
                    metaParts.push(escapeHtml(sizeLabel));
                }
                const createdLabel = formatTimestamp(upload.created_at);
                if (createdLabel) {
                    metaParts.push(escapeHtml(createdLabel));
                }
                const relativePath = escapeHtml(upload.relative_path || "");
                const isAttached = attachedUploadIds.has(upload.id);

                return `
                    <article class="rounded-2xl border border-outline-variant/15 bg-surface-container-high/75 px-4 py-4 shadow-[0_12px_32px_rgba(0,0,0,0.14)]">
                        <div class="flex items-start justify-between gap-3">
                            <div class="min-w-0">
                                <div class="flex items-center gap-2 text-[10px] font-mono uppercase tracking-[0.22em] text-primary/80">
                                    <span class="material-symbols-outlined text-sm">description</span>
                                    <span>用户上传源文件</span>
                                </div>
                                <div class="mt-2 truncate text-sm font-semibold text-on-surface" title="${displayName}">${displayName}</div>
                                ${metaParts.length ? `<div class="mt-1 text-[11px] text-on-surface-variant">${metaParts.join(" · ")}</div>` : ""}
                                ${relativePath ? `<div class="mt-2 break-all text-[11px] font-mono text-on-surface-variant/85">工作区路径：${relativePath}</div>` : ""}
                            </div>
                            <span class="inline-flex shrink-0 items-center rounded-full border border-primary/15 bg-primary/10 px-2.5 py-1 text-[10px] font-mono font-bold tracking-[0.18em] text-primary">源文件</span>
                        </div>
                        <div class="mt-4 flex flex-wrap items-center gap-2">
                            <button
                                type="button"
                                data-upload-action="attach"
                                data-upload-id="${escapeHtml(upload.id)}"
                                class="${isAttached
                                    ? "cursor-default border-emerald-400/20 bg-emerald-400/10 text-emerald-300"
                                    : "hover:border-primary/50 hover:text-primary border-outline-variant/20 bg-surface-container-highest text-on-surface"} inline-flex items-center gap-2 rounded-xl border px-3 py-1.5 text-[11px] font-semibold transition-colors"
                                ${isAttached ? "disabled" : ""}>
                                <span class="material-symbols-outlined text-sm">${isAttached ? "task_alt" : "attach_file"}</span>
                                <span>${isAttached ? "已附加到输入框" : "附加到下一条"}</span>
                            </button>
                            ${renderFileActionButtons(uploadDescriptor)}
                        </div>
                    </article>
                `;
            })
            .join("");
        updateUploadControlsState();
    }

    function renderComposerAttachments() {
        if (!$composerAttachments) {
            renderSessionUploadsPanel();
            updateUploadControlsState();
            return;
        }

        if (!composerAttachments.length) {
            $composerAttachments.classList.add("hidden");
            $composerAttachments.innerHTML = "";
            renderSessionUploadsPanel();
            updateUploadControlsState();
            updateComposerActionButton();
            return;
        }

        const cards = composerAttachments
            .map((attachment) => {
                const upload = attachment.upload || {};
                const displayName = escapeHtml(
                    upload.original_name || upload.safe_name || attachment.file?.name || attachment.name || "未命名文件"
                );
                const sizeLabel = formatFileSize(
                    Number(upload.size_bytes ?? attachment.file?.size ?? attachment.size_bytes)
                );
                const metaParts = [];
                if (upload.mime_type || attachment.file?.type || attachment.mime_type) {
                    metaParts.push(escapeHtml(upload.mime_type || attachment.file?.type || attachment.mime_type));
                }
                if (sizeLabel) {
                    metaParts.push(escapeHtml(sizeLabel));
                }

                let toneClass =
                    "border-primary/20 bg-primary/10 text-primary shadow-[0_10px_30px_rgba(0,238,252,0.08)]";
                if (attachment.status === "ready") {
                    toneClass =
                        "border-emerald-400/20 bg-emerald-400/10 text-emerald-300 shadow-[0_10px_30px_rgba(52,211,153,0.08)]";
                } else if (attachment.status === "error") {
                    toneClass =
                        "border-error/20 bg-error/10 text-on-error-container shadow-[0_10px_30px_rgba(255,113,108,0.08)]";
                }

                const progress =
                    attachment.status === "uploading"
                        ? `
                            <div class="mt-3 h-1.5 overflow-hidden rounded-full bg-surface-container-highest/70">
                                <div class="h-full rounded-full bg-primary transition-[width] duration-200" style="width: ${formatUploadProgress(attachment.progress)};"></div>
                            </div>
                        `
                        : "";

                const removeTitle = attachment.status === "uploading" ? "取消上传" : "从当前消息移除";

                return `
                    <div class="rounded-2xl border px-4 py-3 ${toneClass}">
                        <div class="flex items-start justify-between gap-3">
                            <div class="min-w-0">
                                <div class="flex items-center gap-2 text-[10px] font-mono uppercase tracking-[0.2em]">
                                    <span class="material-symbols-outlined text-sm">${attachment.status === "ready" ? "attach_file" : "upload_file"}</span>
                                    <span>${attachment.source === "session" ? "会话文件" : "新上传文件"}</span>
                                </div>
                                <div class="mt-2 truncate text-sm font-semibold text-on-surface" title="${displayName}">${displayName}</div>
                                ${metaParts.length ? `<div class="mt-1 text-[11px] text-on-surface-variant">${metaParts.join(" · ")}</div>` : ""}
                                <div class="mt-2 text-[11px] font-medium">${escapeHtml(getComposerAttachmentStatusText(attachment))}</div>
                                ${progress}
                            </div>
                            <button
                                type="button"
                                data-remove-composer-attachment="${escapeHtml(attachment.clientId)}"
                                title="${removeTitle}"
                                class="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-white/10 bg-surface-container-highest/60 text-on-surface-variant transition-colors hover:border-white/20 hover:text-on-surface"
                            >
                                <span class="material-symbols-outlined text-sm">close</span>
                            </button>
                        </div>
                    </div>
                `;
            })
            .join("");

        $composerAttachments.classList.remove("hidden");
        $composerAttachments.innerHTML = `
            <div class="mb-2 rounded-2xl border border-outline-variant/10 bg-surface-container-high/50 px-3 py-3">
                <div class="mb-3 flex items-center justify-between gap-3">
                    <div class="text-[10px] font-mono uppercase tracking-[0.22em] text-primary/80">待发送附件</div>
                    <div class="text-[10px] font-mono text-on-surface-variant">${composerAttachments.length} 个文件</div>
                </div>
                <div class="grid gap-2">${cards}</div>
            </div>
        `;
        renderSessionUploadsPanel();
        updateUploadControlsState();
        updateComposerActionButton();
    }

    function clearComposerAttachments() {
        composerAttachments.forEach((attachment) => {
            attachment.discarded = true;
            if (attachment.xhr && attachment.status === "uploading") {
                try {
                    attachment.xhr.abort();
                } catch (error) {
                    console.warn("Failed to abort upload:", error);
                }
            }
        });
        composerAttachments = [];
        renderComposerAttachments();
    }

    function removeComposerAttachment(clientId) {
        const target = composerAttachments.find((attachment) => attachment.clientId === clientId);
        if (!target) {
            return;
        }
        target.discarded = true;
        if (target.xhr && target.status === "uploading") {
            try {
                target.xhr.abort();
            } catch (error) {
                console.warn("Failed to abort upload:", error);
            }
        }
        composerAttachments = composerAttachments.filter((attachment) => attachment.clientId !== clientId);
        renderComposerAttachments();
    }

    function addUploadToComposer(upload, source = "session") {
        const normalized = normalizeUploadRecord(upload);
        if (!normalized) {
            return;
        }
        const alreadyAttached = composerAttachments.some(
            (attachment) => attachment.status === "ready" && attachment.upload?.id === normalized.id
        );
        if (alreadyAttached) {
            return;
        }

        composerAttachments = [
            ...composerAttachments,
            {
                clientId: generateClientId("attached-upload"),
                source,
                status: "ready",
                progress: 100,
                errorMessage: "",
                discarded: false,
                upload: normalized,
            },
        ];
        renderComposerAttachments();
    }

    async function ensureSessionForUploads() {
        if (sessionId) {
            return sessionId;
        }
        const created = await createSession({ openChat: true });
        return created?.session_id || sessionId;
    }

    function getXhrErrorMessage(xhr) {
        if (xhr?.response && typeof xhr.response === "object" && typeof xhr.response.detail === "string") {
            return xhr.response.detail;
        }
        if (typeof xhr?.responseText === "string" && xhr.responseText.trim()) {
            try {
                const parsed = JSON.parse(xhr.responseText);
                if (typeof parsed?.detail === "string" && parsed.detail) {
                    return parsed.detail;
                }
            } catch {
                // Ignore invalid JSON payloads and fall back to the status text.
            }
        }
        return xhr?.statusText || `HTTP ${xhr?.status || 0}`;
    }

    function uploadFileToSession(targetSessionId, attachment) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            attachment.xhr = xhr;
            xhr.open("POST", `/api/sessions/${encodeURIComponent(targetSessionId)}/uploads`);
            xhr.responseType = "json";

            xhr.upload.addEventListener("progress", (event) => {
                if (!event.lengthComputable) {
                    return;
                }
                attachment.progress = Math.round((event.loaded / event.total) * 100);
                if (targetSessionId === sessionId) {
                    renderComposerAttachments();
                }
            });

            xhr.addEventListener("load", () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(xhr.response);
                    return;
                }
                reject(new Error(getXhrErrorMessage(xhr)));
            });
            xhr.addEventListener("error", () => {
                reject(new Error("上传失败，请检查网络后重试。"));
            });
            xhr.addEventListener("abort", () => {
                const error = new Error("上传已取消");
                error.name = "AbortError";
                reject(error);
            });

            const formData = new FormData();
            formData.append("files", attachment.file, attachment.file.name);
            xhr.send(formData);
        });
    }

    async function handleSelectedFiles(fileList) {
        const files = Array.from(fileList || []).filter((file) => file instanceof File);
        if (!files.length) {
            return;
        }
        if (isStreaming) {
            setStatus("当前任务执行中，暂不支持新增上传");
            updateUploadControlsState();
            return;
        }

        const targetSessionId = await ensureSessionForUploads();
        if (!targetSessionId) {
            return;
        }

        showChatView();

        const queuedAttachments = files.map((file) => ({
            clientId: generateClientId("pending-upload"),
            source: "local",
            status: "uploading",
            progress: 0,
            errorMessage: "",
            discarded: false,
            file,
            name: file.name,
            size_bytes: file.size,
            mime_type: file.type || "",
            upload: null,
            xhr: null,
        }));

        composerAttachments = [...composerAttachments, ...queuedAttachments];
        renderComposerAttachments();
        setStatus(`正在上传 ${files.length} 个文件`);

        await Promise.all(
            queuedAttachments.map(async (attachment) => {
                try {
                    const payload = await uploadFileToSession(targetSessionId, attachment);
                    const upload = normalizeUploadRecord(Array.isArray(payload?.uploads) ? payload.uploads[0] : null);
                    if (upload && targetSessionId === sessionId) {
                        upsertSessionUploads([upload]);
                    }
                    if (attachment.discarded) {
                        return;
                    }
                    if (!upload) {
                        throw new Error("上传成功，但返回结果缺少文件信息。");
                    }
                    attachment.status = "ready";
                    attachment.progress = 100;
                    attachment.errorMessage = "";
                    attachment.upload = upload;
                } catch (error) {
                    if (attachment.discarded || error?.name === "AbortError") {
                        return;
                    }
                    attachment.status = "error";
                    attachment.progress = 0;
                    attachment.errorMessage = error?.message || "上传失败";
                } finally {
                    attachment.xhr = null;
                    if (targetSessionId === sessionId) {
                        renderComposerAttachments();
                    }
                }
            })
        );

        if (targetSessionId !== sessionId) {
            return;
        }

        const readyCount = getReadyComposerAttachments().length;
        const failedCount = composerAttachments.filter((attachment) => attachment.status === "error").length;
        if (failedCount > 0) {
            setStatus(`已有 ${readyCount} 个附件就绪，${failedCount} 个上传失败`);
            return;
        }
        setStatus(readyCount > 0 ? `已附加 ${readyCount} 个文件` : "上传已取消");
    }

    async function refreshSessionUploads(targetSessionId = sessionId) {
        if (!targetSessionId) {
            clearSessionUploadsPanel();
            return;
        }

        try {
            const data = await fetchJson(`/api/sessions/${targetSessionId}/uploads`);
            if (targetSessionId !== sessionId) {
                return;
            }
            sessionUploads = sortUploadsByCreatedAt(Array.isArray(data.uploads) ? data.uploads : []);
            renderSessionUploadsPanel();
        } catch (error) {
            console.error("Failed to load session uploads:", error);
            if (targetSessionId === sessionId) {
                clearSessionUploadsPanel();
            }
        }
    }

    function buildOutgoingUserContent(text, attachments) {
        const normalizedText = (text || "").trim();
        const blocks = [];

        if (normalizedText) {
            blocks.push({ type: "text", text: normalizedText });
        }
        attachments.forEach((attachment) => {
            if (attachment.upload?.id) {
                blocks.push(createUploadedFileBlock(attachment.upload));
            }
        });

        if (blocks.length === 1 && blocks[0].type === "text") {
            return normalizedText;
        }
        return blocks;
    }

    function structuredMessageSummary(content) {
        const blocks = normalizeMessageBlocks(content);
        if (!blocks.length) {
            return "结构化消息";
        }

        const textParts = [];
        const uploadNames = [];
        const artifactNames = [];

        blocks.forEach((block) => {
            if (block.type === "text" && typeof block.text === "string" && block.text.trim()) {
                textParts.push(block.text.trim().replace(/\s+/g, " "));
                return;
            }

            if (block.type === "uploaded_file") {
                const name = (block.original_name || block.safe_name || block.upload_id || "").trim();
                if (name) {
                    uploadNames.push(name);
                }
                return;
            }

            if (block.type === "artifact_ref") {
                const name = (block.display_name || block.uri || block.artifact_id || "").trim();
                if (name) {
                    artifactNames.push(name);
                }
            }
        });

        const parts = [];
        if (textParts.length) {
            parts.push(textParts.join(" "));
        }
        if (uploadNames.length) {
            parts.push(`已附带上传文件：${uploadNames.slice(0, 3).join("、")}${uploadNames.length > 3 ? " 等" : ""}`);
        }
        if (artifactNames.length) {
            parts.push(`已引用产物：${artifactNames.slice(0, 3).join("、")}${artifactNames.length > 3 ? " 等" : ""}`);
        }

        return parts.join(" ") || "结构化消息";
    }

    function messageText(content) {
        if (typeof content === "string") return content;
        return structuredMessageSummary(content);
    }

    function formatTimestamp(value) {
        if (!value) return "";
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return value;
        return date.toLocaleString("zh-CN", {
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
        });
    }

    function formatDurationMs(durationMs) {
        if (!Number.isFinite(durationMs) || durationMs < 0) {
            return "0s";
        }
        const totalSeconds = Math.round(durationMs / 1000);
        if (totalSeconds < 60) {
            return `${totalSeconds}s`;
        }
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;
        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        }
        if (minutes > 0 && seconds === 0) {
            return `${minutes}m`;
        }
        return `${minutes}m ${seconds}s`;
    }

    function computeRunDurationMs(run) {
        if (!run?.started_at) {
            return 0;
        }
        const startedAt = new Date(run.started_at).getTime();
        if (Number.isNaN(startedAt)) {
            return 0;
        }
        const finishedAt = run.finished_at ? new Date(run.finished_at).getTime() : Date.now();
        if (Number.isNaN(finishedAt)) {
            return 0;
        }
        return Math.max(0, finishedAt - startedAt);
    }

    function getRunStatusMeta(status) {
        const normalized = (status || "").toLowerCase();
        const statusMap = {
            queued: {
                label: "排队中",
                icon: "schedule",
                tone: "text-sky-300 border-sky-400/20 bg-sky-400/10",
            },
            running: {
                label: "执行中",
                icon: "progress_activity",
                tone: "text-primary border-primary/20 bg-primary/10",
            },
            waiting_approval: {
                label: "待确认",
                icon: "approval",
                tone: "text-amber-200 border-amber-400/20 bg-amber-400/10",
            },
            interrupted: {
                label: "已中断",
                icon: "pause_circle",
                tone: "text-orange-200 border-orange-400/20 bg-orange-400/10",
            },
            completed: {
                label: "已完成",
                icon: "task_alt",
                tone: "text-emerald-300 border-emerald-400/20 bg-emerald-400/10",
            },
            failed: {
                label: "已失败",
                icon: "error",
                tone: "text-error border-error/20 bg-error/10",
            },
            timed_out: {
                label: "已超时",
                icon: "timer_off",
                tone: "text-rose-300 border-rose-400/20 bg-rose-400/10",
            },
            cancelled: {
                label: "已取消",
                icon: "do_not_disturb_on",
                tone: "text-on-surface-variant border-outline-variant/20 bg-surface-container-high",
            },
        };
        return statusMap[normalized] || {
            label: normalized || "未知",
            icon: "help",
            tone: "text-on-surface-variant border-outline-variant/20 bg-surface-container-high",
        };
    }

    function isRunActive(run) {
        return Boolean(run && RUN_ACTIVE_STATUSES.has(run.status));
    }

    function isRunRecoverable(run) {
        return run?.status === "interrupted";
    }

    function getRootRuns() {
        return sessionRuns.filter((run) => !run.parent_run_id);
    }

    function getRunChildren(runId) {
        return sessionRuns.filter((run) => run.parent_run_id === runId);
    }

    function getCurrentRootRun() {
        return getRootRuns().find((run) => isRunActive(run)) || null;
    }

    function getSelectedRun() {
        return sessionRuns.find((run) => run.id === selectedRunId) || null;
    }

    function getRunById(runId) {
        return sessionRuns.find((run) => run.id === runId) || null;
    }

    function resolveRootRunId(runId) {
        const normalizedRunId = String(runId || "").trim();
        if (!normalizedRunId) {
            return "";
        }
        let current = getRunById(normalizedRunId);
        if (!current) {
            return normalizedRunId;
        }
        const visited = new Set();
        while (current?.parent_run_id && !visited.has(current.id)) {
            visited.add(current.id);
            current = getRunById(current.parent_run_id);
        }
        return String(current?.id || normalizedRunId).trim();
    }

    function getRunAgentName(run) {
        if (!run) return "未知";
        return (
            run.run_metadata?.agent_name ||
            run.agent_template_snapshot?.name ||
            run.agent_template_id ||
            "未知"
        );
    }

    function buildRunFocusSummary(run, { isLive = false, pendingCount = 0 } = {}) {
        if (!run) {
            return "先输入一个明确任务，开始后这里会显示进展。";
        }
        if (pendingCount > 0) {
            return `当前有 ${pendingCount} 个请求等待处理，完成后任务才会继续。`;
        }
        if (run.status === "running" || run.status === "queued") {
            return isLive
                ? "智能体正在持续处理当前任务。"
                : "这次执行还在进行中。";
        }
        if (run.status === "waiting_approval") {
            return "当前任务已暂停，等待你确认是否继续。";
        }
        if (run.status === "interrupted") {
            return "当前任务在完成前中断了，你可以在详情面板中继续执行。";
        }
        if (run.status === "completed") {
            return "当前任务已经完成；只有在需要完整过程时才需要查看详细信息。";
        }
        if (run.status === "failed") {
            return run.error_summary
                ? truncate(run.error_summary, 140)
                : "当前任务在完成前失败了。";
        }
        return "如果你需要更详细的过程，可以打开详情面板。";
    }

    function buildRunNextAction(run, { pendingCount = 0 } = {}) {
        if (!run) {
            return "在下方输入任务即可开始。";
        }
        if (pendingCount > 0 || run.status === "waiting_approval") {
            return "先处理上方的待确认请求，或在下方补充说明。";
        }
        if (run.status === "running" || run.status === "queued") {
            return "可以先让智能体继续执行；如果想调整方向，也可以在下方补充要求。";
        }
        if (run.status === "interrupted") {
            return "如果想从中断处继续，可以在详情面板里继续执行。";
        }
        if (run.status === "failed") {
            return "如果要排查原因，可查看详细过程；也可以直接修改要求后重新尝试。";
        }
        return "你可以在下方继续迭代结果，或者开始下一个任务。";
    }

    function buildSessionStatusSnapshot() {
        const rootRuns = getRootRuns();
        const activeRun = getCurrentRootRun();
        const latestRun = activeRun || rootRuns[0] || null;

        if (!latestRun) {
            return {
                eyebrow: "准备就绪",
                title: "当前还没有开始任务",
                body: "输入一个明确任务后，这里会显示进展和需要你处理的事项。",
                badgeLabel: "空闲",
                badgeTone: "text-on-surface-variant border-outline-variant/20 bg-surface-container-high",
            };
        }

        if (sessionPendingApprovals.length > 0) {
            return {
                eyebrow: "需要你处理",
                title: `当前有 ${sessionPendingApprovals.length} 个请求等待处理`,
                body: "任务目前被阻塞，请先在主区域处理请求，然后它才会继续执行。",
                badgeLabel: "待处理",
                badgeTone: "text-amber-200 border-amber-400/20 bg-amber-400/10",
            };
        }

        const meta = getRunStatusMeta(latestRun.status);
        return {
            eyebrow: activeRun ? "当前任务" : "最新任务",
            title: latestRun.goal || "未命名任务",
            body: `${buildRunFocusSummary(latestRun, { isLive: Boolean(activeRun) })} ${buildRunMetaLine(latestRun)}`,
            badgeLabel: meta.label,
            badgeTone: meta.tone,
        };
    }

    function getApprovalRiskMeta(riskLevel) {
        const normalized = (riskLevel || "").toLowerCase();
        const tones = {
            low: "text-emerald-300 border-emerald-400/20 bg-emerald-400/10",
            medium: "text-sky-300 border-sky-400/20 bg-sky-400/10",
            high: "text-amber-200 border-amber-400/20 bg-amber-400/10",
            critical: "text-error border-error/20 bg-error/10",
        };
        return {
            label:
                {
                    low: "低风险",
                    medium: "中风险",
                    high: "高风险",
                    critical: "极高风险",
                }[normalized] || "未知风险",
            tone: tones[normalized] || "text-on-surface-variant border-outline-variant/20 bg-surface-container-high",
        };
    }

    function getApprovalDecisionScopeMeta(scope) {
        const normalized = (scope || "").toLowerCase();
        const scopes = {
            once: {
                label: "仅本次",
                tone: "text-primary border-primary/20 bg-primary/10",
            },
            run: {
                label: "本次任务",
                tone: "text-sky-300 border-sky-400/20 bg-sky-400/10",
            },
            template: {
                label: "已更新规则",
                tone: "text-emerald-300 border-emerald-400/20 bg-emerald-400/10",
            },
        };
        return scopes[normalized] || null;
    }

    function isApprovalPending(approval) {
        return approval?.status === "pending";
    }

    function isApprovalActionPending(approvalId) {
        return approvalActionIds.has(approvalId);
    }

    function buildApprovalMetaLine(approval) {
        const parts = [];
        const relatedRun = getRunById(approval?.run_id);
        if (relatedRun?.goal) {
            parts.push(`任务 ${relatedRun.goal}`);
        } else if (approval?.run_id) {
            parts.push(`任务 ${approval.run_id.slice(0, 8)}`);
        }
        if (approval?.requested_at) {
            parts.push(`发起于 ${formatTimestamp(approval.requested_at)}`);
        }
        if (approval?.resolved_at) {
            parts.push(`处理于 ${formatTimestamp(approval.resolved_at)}`);
        }
        return parts.join(" / ");
    }

    function renderApprovalActions(approval, { compact = false } = {}) {
        if (!isApprovalPending(approval)) {
            return "";
        }

        const pending = isApprovalActionPending(approval.id);
        const disabledAttr = pending ? "disabled aria-disabled=\"true\"" : "";
        const commonClass = pending ? "opacity-60 cursor-not-allowed" : "";
        const paddingClass = compact ? "px-2.5 py-2" : "px-3 py-2";

        return `
            <div class="mt-3 flex flex-wrap gap-2">
                <button
                    type="button"
                    data-approval-action="grant-once"
                    data-approval-id="${escapeHtml(approval.id)}"
                    data-run-id="${escapeHtml(approval.run_id || "")}"
                    ${disabledAttr}
                    class="inline-flex items-center gap-2 rounded-xl border border-primary/30 bg-primary/10 ${paddingClass} text-[10px] font-mono uppercase tracking-[0.18em] text-primary hover:bg-primary/15 transition-colors ${commonClass}"
                >
                    <span class="material-symbols-outlined text-sm">${pending ? "progress_activity" : "verified"}</span>
                    <span>${pending ? "处理中" : "仅同意这次"}</span>
                </button>
                <button
                    type="button"
                    data-approval-action="grant-run"
                    data-approval-id="${escapeHtml(approval.id)}"
                    data-run-id="${escapeHtml(approval.run_id || "")}"
                    ${disabledAttr}
                    class="inline-flex items-center gap-2 rounded-xl border border-sky-400/20 bg-sky-400/10 ${paddingClass} text-[10px] font-mono uppercase tracking-[0.18em] text-sky-300 hover:bg-sky-400/15 transition-colors ${commonClass}"
                >
                    <span class="material-symbols-outlined text-sm">resume</span>
                    <span>${compact ? "本次任务" : "同意本次任务"}</span>
                </button>
                <button
                    type="button"
                    data-approval-action="grant-template"
                    data-approval-id="${escapeHtml(approval.id)}"
                    data-run-id="${escapeHtml(approval.run_id || "")}"
                    ${disabledAttr}
                    class="inline-flex items-center gap-2 rounded-xl border border-emerald-400/20 bg-emerald-400/10 ${paddingClass} text-[10px] font-mono uppercase tracking-[0.18em] text-emerald-300 hover:bg-emerald-400/15 transition-colors ${commonClass}"
                >
                    <span class="material-symbols-outlined text-sm">rule_settings</span>
                    <span>${compact ? "规则" : "更新规则"}</span>
                </button>
                <button
                    type="button"
                    data-approval-action="deny"
                    data-approval-id="${escapeHtml(approval.id)}"
                    data-run-id="${escapeHtml(approval.run_id || "")}"
                    ${disabledAttr}
                    class="inline-flex items-center gap-2 rounded-xl border border-error/20 bg-error/10 ${paddingClass} text-[10px] font-mono uppercase tracking-[0.18em] text-error hover:bg-error/15 transition-colors ${commonClass}"
                >
                    <span class="material-symbols-outlined text-sm">block</span>
                    <span>拒绝</span>
                </button>
            </div>
        `;
    }

    function renderApprovalRecord(approval, { compact = false } = {}) {
        const approvalMeta = getRunStatusMeta(
            approval.status === "pending"
                ? "waiting_approval"
                : approval.status === "granted"
                    ? "completed"
                    : "failed"
        );
        const riskMeta = getApprovalRiskMeta(approval.risk_level);
        const scopeMeta = getApprovalDecisionScopeMeta(approval.decision_scope);
        const relatedRun = getRunById(approval.run_id);
        const inspectLabel = relatedRun?.goal || approval.run_id?.slice(0, 8) || "查看任务";
        const approvalStatusLabel =
            approval.status === "pending"
                ? "待处理"
                : approval.status === "granted"
                    ? "已同意"
                    : approval.status === "denied"
                        ? "已拒绝"
                        : approval.status || "未知";
        const paddingClass = compact ? "px-3 py-3" : "px-4 py-4";

        return `
            <article class="rounded-2xl border border-outline-variant/10 bg-surface-container-low ${paddingClass}">
                <div class="flex items-start justify-between gap-3">
                    <div class="min-w-0">
                        <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">${escapeHtml(approval.tool_name || "审批")}</div>
                        <div class="mt-1 text-sm font-bold text-on-surface">${escapeHtml(approval.parameter_summary || approval.impact_summary || "审批请求")}</div>
                        <div class="mt-2 text-[11px] text-on-surface-variant">${escapeHtml(buildApprovalMetaLine(approval) || "等待处理")}</div>
                    </div>
                    <div class="flex flex-col items-end gap-2 shrink-0">
                        <span class="inline-flex items-center rounded-full border px-2 py-1 text-[10px] font-mono uppercase tracking-[0.18em] ${approvalMeta.tone}">${escapeHtml(approvalStatusLabel)}</span>
                        <span class="inline-flex items-center rounded-full border px-2 py-1 text-[10px] font-mono uppercase tracking-[0.18em] ${riskMeta.tone}">${escapeHtml(riskMeta.label)}</span>
                        ${
                            scopeMeta
                                ? `<span class="inline-flex items-center rounded-full border px-2 py-1 text-[10px] font-mono uppercase tracking-[0.18em] ${scopeMeta.tone}">${escapeHtml(scopeMeta.label)}</span>`
                                : ""
                        }
                    </div>
                </div>
                <div class="mt-3 grid gap-2">
                    <div class="rounded-xl border border-outline-variant/10 bg-black/20 px-3 py-3">
                        <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">请求内容</div>
                        <div class="mt-2 text-[11px] leading-relaxed text-on-surface">${escapeHtml(approval.parameter_summary || "暂无请求内容说明。")}</div>
                    </div>
                    <div class="rounded-xl border border-outline-variant/10 bg-black/20 px-3 py-3">
                        <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">影响范围</div>
                        <div class="mt-2 text-[11px] leading-relaxed text-on-surface">${escapeHtml(approval.impact_summary || "暂无影响范围说明。")}</div>
                    </div>
                    <div class="rounded-xl border border-outline-variant/10 bg-black/20 px-3 py-3">
                        <div class="flex items-center justify-between gap-3">
                            <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">关联任务</div>
                            ${
                                approval.run_id
                                    ? `<button type="button" data-run-action="select" data-run-id="${escapeHtml(approval.run_id)}" class="inline-flex items-center gap-1 text-[10px] font-mono uppercase tracking-[0.18em] text-primary hover:text-primary-dim transition-colors">
                                        <span>${escapeHtml(truncate(inspectLabel, compact ? 24 : 36))}</span>
                                        <span class="material-symbols-outlined text-sm">arrow_outward</span>
                                       </button>`
                                    : ""
                            }
                        </div>
                    </div>
                </div>
                ${
                    approval.decision_notes
                        ? `<div class="mt-3 rounded-xl border border-outline-variant/10 bg-surface-container-high px-3 py-3 text-[11px] text-on-surface-variant">
                            <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">处理说明</div>
                            <div class="mt-2 leading-relaxed">${escapeHtml(approval.decision_notes)}</div>
                           </div>`
                        : ""
                }
                ${renderApprovalActions(approval, { compact })}
            </article>
        `;
    }

    function buildRunMetaLine(run) {
        const parts = [];
        if (run?.started_at) {
            parts.push(`开始于 ${formatTimestamp(run.started_at)}`);
        } else {
            parts.push(`创建于 ${formatTimestamp(run?.created_at)}`);
        }
        parts.push(`耗时 ${formatDurationMs(computeRunDurationMs(run))}`);
        if (Number.isFinite(run?.current_step_index)) {
            parts.push(`步骤 #${run.current_step_index}`);
        }
        return parts.join(" / ");
    }

    function renderInspectorSection(title, content, { open = false } = {}) {
        return `
            <details class="rounded-2xl border border-outline-variant/10 bg-surface-container-low" ${open ? "open" : ""}>
                <summary class="cursor-pointer list-none px-3 py-3 text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">
                    ${escapeHtml(title)}
                </summary>
                <div class="px-3 pb-3">
                    ${content}
                </div>
            </details>
        `;
    }

    function renderRunDagSection() {
        if (!isFeatureEnabled("enable_run_trace")) {
            return `<div class="rounded-xl border border-dashed border-outline-variant/20 bg-surface-container-high px-3 py-3 text-sm text-on-surface-variant">当前环境未开启子任务树功能。</div>`;
        }
        if (!selectedRunTree) {
            return `<div class="rounded-xl border border-dashed border-outline-variant/20 bg-surface-container-high px-3 py-3 text-sm text-on-surface-variant">当前没有可查看的子任务图谱。</div>`;
        }
        if (window.RunDagView?.renderSection) {
            return window.RunDagView.renderSection({
                tree: selectedRunTree,
                selectedRunId,
            });
        }
        return `<div class="rounded-xl border border-dashed border-outline-variant/20 bg-surface-container-high px-3 py-3 text-sm text-on-surface-variant">子任务 DAG 组件加载失败。</div>`;
    }

    function setSidebarTab(tabName) {
        const durableRunsEnabled = isFeatureEnabled("enable_durable_runs");
        const allowedTabs = durableRunsEnabled
            ? new Set(["runs", "context", "learning"])
            : new Set(["context", "learning"]);
        const normalizedTab = allowedTabs.has(tabName) ? tabName : durableRunsEnabled ? "runs" : "context";
        activeSidebarTab = normalizedTab;
        $sidebarTabButtons.forEach((button) => {
            const active = button.dataset.sidebarTab === activeSidebarTab;
            button.classList.toggle("text-primary", active);
            button.classList.toggle("border-primary", active);
            button.classList.toggle("text-on-surface-variant", !active);
            button.classList.toggle("border-transparent", !active);
        });
        if ($runsPanel) {
            $runsPanel.classList.toggle("hidden", !durableRunsEnabled || activeSidebarTab !== "runs");
        }
        if ($contextPanel) {
            $contextPanel.classList.toggle("hidden", activeSidebarTab !== "context");
        }
        if ($learningPanel) {
            $learningPanel.classList.toggle("hidden", activeSidebarTab !== "learning");
        }
    }

    function formatProfileFieldLabel(fieldName) {
        const labels = {
            preferred_language: "偏好语言",
            response_length: "回复长度",
            technical_depth: "技术深度",
            recurring_projects: "长期项目",
            dislikes_avoidances: "避讳与偏好",
            approval_risk_preference: "风险偏好",
            timezone: "时区",
            locale: "地区设置",
        };
        return labels[fieldName] || fieldName;
    }

    function formatProfileFieldValue(value) {
        if (Array.isArray(value)) {
            return value.length ? value.join("、") : "未设置";
        }
        if (value && typeof value === "object") {
            return JSON.stringify(value, null, 2);
        }
        const normalized = String(value ?? "").trim();
        return normalized || "未设置";
    }

    function formatMemoryTypeLabel(memoryType) {
        const labels = {
            preference: "偏好",
            communication_style: "沟通风格",
            goal: "目标",
            constraint: "约束",
            project_fact: "项目事实",
            workflow_fact: "流程经验",
            correction: "纠正",
        };
        return labels[memoryType] || memoryType || "长期记忆";
    }

    function formatWriterLabel(writerType, writerId) {
        const typeLabel = String(writerType || "").trim() || "system";
        const idLabel = String(writerId || "").trim();
        return idLabel ? `${typeLabel}:${idLabel}` : typeLabel;
    }

    function getRunStartedEventForSelectedRun() {
        if (!Array.isArray(selectedRunTimeline) || !selectedRunTimeline.length) {
            return null;
        }
        const normalizedSelectedRunId = String(selectedRunId || "").trim();
        const matchingRunStarted = [...selectedRunTimeline]
            .reverse()
            .find((item) => item?.event_type === "run_started" && String(item?.run_id || "").trim() === normalizedSelectedRunId);
        if (matchingRunStarted) {
            return matchingRunStarted;
        }
        return [...selectedRunTimeline]
            .reverse()
            .find((item) => item?.event_type === "run_started") || null;
    }

    function getPromptMemorySectionsForSelectedRun() {
        const runStarted = getRunStartedEventForSelectedRun();
        const sections = runStarted?.data?.prompt?.memory_sections;
        return Array.isArray(sections) ? sections : [];
    }

    function getRetrievedContextSection() {
        return getPromptMemorySectionsForSelectedRun().find((section) => section?.key === "retrieved_context") || null;
    }

    function renderLearningMetricCard(label, value, tone = "cyan") {
        const tones = {
            cyan: "border-primary/20 bg-primary/10 text-primary",
            violet: "border-secondary/20 bg-secondary/10 text-secondary",
            slate: "border-outline-variant/20 bg-surface-container-high text-on-surface",
            emerald: "border-emerald-400/20 bg-emerald-400/10 text-emerald-300",
        };
        return `
            <article class="rounded-2xl border px-4 py-4 ${tones[tone] || tones.slate}">
                <div class="text-[10px] font-mono uppercase tracking-[0.18em] opacity-80">${escapeHtml(label)}</div>
                <div class="mt-2 text-xl font-semibold">${escapeHtml(String(value))}</div>
            </article>
        `;
    }

    function renderLearningEmpty(message) {
        return `
            <div class="rounded-2xl border border-dashed border-outline-variant/20 bg-surface-container-high px-4 py-4 text-sm leading-relaxed text-on-surface-variant">
                ${escapeHtml(message)}
            </div>
        `;
    }

    function renderProfileFieldCard(fieldName, value, meta = {}) {
        const provenance = [
            meta?.source ? `来源：${meta.source}` : "",
            Number.isFinite(meta?.confidence) ? `置信度：${Math.round(Number(meta.confidence) * 100)}%` : "",
            meta?.source_session_id ? `会话：${meta.source_session_id.slice(0, 8)}` : "",
            meta?.source_run_id ? `运行：${meta.source_run_id.slice(0, 8)}` : "",
            meta?.updated_at ? `更新：${formatTimestamp(meta.updated_at)}` : "",
        ].filter(Boolean);
        return `
            <article class="rounded-2xl border border-outline-variant/15 bg-surface-container-high/60 px-4 py-4">
                <div class="flex items-start justify-between gap-3">
                    <div class="min-w-0">
                        <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-primary/80">${escapeHtml(formatProfileFieldLabel(fieldName))}</div>
                        <div class="mt-2 text-sm font-semibold text-on-surface whitespace-pre-wrap break-words">${escapeHtml(formatProfileFieldValue(value))}</div>
                        <div class="mt-2 text-[11px] leading-relaxed text-on-surface-variant">${escapeHtml(provenance.join(" · ") || "暂无溯源信息")}</div>
                        <div class="mt-1 text-[11px] text-on-surface-variant">写入方：${escapeHtml(formatWriterLabel(meta?.writer_type, meta?.writer_id))}</div>
                    </div>
                    <div class="flex shrink-0 items-center gap-2">
                        <button
                            type="button"
                            data-memory-action="edit-profile-field"
                            data-field-name="${escapeHtml(fieldName)}"
                            class="inline-flex items-center gap-1 rounded-full border border-outline-variant/20 bg-surface-container-highest px-2.5 py-1 text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant hover:border-primary/40 hover:text-primary transition-colors"
                        >
                            <span class="material-symbols-outlined text-sm">edit</span>
                            <span>修正</span>
                        </button>
                        <button
                            type="button"
                            data-memory-action="remove-profile-field"
                            data-field-name="${escapeHtml(fieldName)}"
                            class="inline-flex items-center gap-1 rounded-full border border-error/20 bg-error/10 px-2.5 py-1 text-[10px] font-mono uppercase tracking-[0.18em] text-error hover:bg-error/15 transition-colors"
                        >
                            <span class="material-symbols-outlined text-sm">delete</span>
                            <span>清除</span>
                        </button>
                    </div>
                </div>
            </article>
        `;
    }

    function renderMemoryEntryCard(entry) {
        const provenance = [
            `类型：${formatMemoryTypeLabel(entry.memory_type)}`,
            `置信度：${Math.round(Number(entry.confidence || 0) * 100)}%`,
            entry.source_session_id ? `会话：${entry.source_session_id.slice(0, 8)}` : "",
            entry.source_run_id ? `运行：${entry.source_run_id.slice(0, 8)}` : "",
            entry.updated_at ? `更新：${formatTimestamp(entry.updated_at)}` : "",
        ].filter(Boolean);
        return `
            <article class="rounded-2xl border border-outline-variant/15 bg-surface-container-high/60 px-4 py-4">
                <div class="flex items-start justify-between gap-3">
                    <div class="min-w-0">
                        <div class="flex items-center gap-2">
                            <span class="inline-flex items-center rounded-full border border-primary/20 bg-primary/10 px-2 py-1 text-[10px] font-mono uppercase tracking-[0.18em] text-primary">${escapeHtml(formatMemoryTypeLabel(entry.memory_type))}</span>
                            <span class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">${escapeHtml(entry.id.slice(0, 8))}</span>
                        </div>
                        <div class="mt-2 text-sm font-semibold text-on-surface">${escapeHtml(entry.summary || "未提供摘要")}</div>
                        <div class="mt-2 whitespace-pre-wrap break-words text-[12px] leading-relaxed text-on-surface-variant">${escapeHtml(entry.content || "")}</div>
                        <div class="mt-3 text-[11px] leading-relaxed text-on-surface-variant">${escapeHtml(provenance.join(" · "))}</div>
                        <div class="mt-1 text-[11px] text-on-surface-variant">写入方：${escapeHtml(formatWriterLabel(entry.writer_type, entry.writer_id))}</div>
                    </div>
                    <div class="flex shrink-0 items-center gap-2">
                        <button
                            type="button"
                            data-memory-action="edit-memory-entry"
                            data-entry-id="${escapeHtml(entry.id)}"
                            class="inline-flex items-center gap-1 rounded-full border border-outline-variant/20 bg-surface-container-highest px-2.5 py-1 text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant hover:border-primary/40 hover:text-primary transition-colors"
                        >
                            <span class="material-symbols-outlined text-sm">edit_note</span>
                            <span>修正</span>
                        </button>
                        <button
                            type="button"
                            data-memory-action="delete-memory-entry"
                            data-entry-id="${escapeHtml(entry.id)}"
                            class="inline-flex items-center gap-1 rounded-full border border-error/20 bg-error/10 px-2.5 py-1 text-[10px] font-mono uppercase tracking-[0.18em] text-error hover:bg-error/15 transition-colors"
                        >
                            <span class="material-symbols-outlined text-sm">delete</span>
                            <span>删除</span>
                        </button>
                    </div>
                </div>
            </article>
        `;
    }

    function renderPromptSectionCard(section) {
        const sources = Array.isArray(section?.sources) ? section.sources : [];
        const sourceHtml = sources.length
            ? `<div class="mt-3 flex flex-wrap gap-2">${sources
                  .map(
                      (item) => `
                        <span class="inline-flex items-center rounded-full border border-outline-variant/20 bg-surface-container-highest px-2 py-1 text-[10px] font-mono text-on-surface-variant">${escapeHtml(String(item))}</span>
                      `
                  )
                  .join("")}</div>`
            : `<div class="mt-3 text-[11px] text-on-surface-variant">当前没有可展示的来源信息。</div>`;
        const body = String(section?.body || "").trim();
        const bodyHtml = body
            ? `<div class="markdown mt-3 text-sm leading-relaxed text-on-surface">${renderMarkdownSafe(body)}</div>`
            : `
                <div class="mt-3 rounded-2xl border border-dashed border-outline-variant/20 bg-surface-container-highest/60 px-4 py-3 text-[12px] leading-relaxed text-on-surface-variant">
                    当前 trace 没有保存这段注入正文。若这是修复前生成的旧运行，请重新执行一次后再查看。
                </div>
            `;
        return `
            <article class="rounded-2xl border border-outline-variant/15 bg-surface-container-high/60 px-4 py-4">
                <div class="flex items-start justify-between gap-3">
                    <div class="min-w-0">
                        <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-primary/80">${escapeHtml(section?.title || "注入上下文")}</div>
                        <div class="mt-2 text-[11px] text-on-surface-variant">条目：${escapeHtml(String(section?.items || 0))} · 字符：${escapeHtml(String(section?.chars || 0))}</div>
                    </div>
                </div>
                ${bodyHtml}
                ${sourceHtml}
            </article>
        `;
    }

    function renderWorkflowCandidateCard(item) {
        const statusMeta = {
            pending_review: "待审核",
            approved: "已批准",
            rejected: "已拒绝",
            installed: "已安装",
        };
        const provenance = [
            item.run_id ? `运行：${item.run_id.slice(0, 8)}` : "",
            item.agent_template_id ? `Agent：${getAgentDisplayName(item.agent_template_id)}` : "",
            Array.isArray(item.signal_types) && item.signal_types.length
                ? `信号：${item.signal_types.join("、")}`
                : "",
            item.updated_at ? `更新：${formatTimestamp(item.updated_at)}` : "",
        ].filter(Boolean);
        const skillPreview = String(item.generated_skill_markdown || "").trim();
        return `
            <details class="rounded-2xl border border-outline-variant/15 bg-surface-container-high/60 px-4 py-4">
                <summary class="cursor-pointer list-none">
                    <div class="flex items-start justify-between gap-3">
                        <div class="min-w-0">
                            <div class="flex items-center gap-2">
                                <span class="inline-flex items-center rounded-full border border-secondary/20 bg-secondary/10 px-2 py-1 text-[10px] font-mono uppercase tracking-[0.18em] text-secondary">${escapeHtml(statusMeta[item.status] || item.status || "候选")}</span>
                                <span class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">${escapeHtml(item.id.slice(0, 8))}</span>
                            </div>
                            <div class="mt-2 text-sm font-semibold text-on-surface">${escapeHtml(item.title || item.suggested_skill_name || "未命名流程候选")}</div>
                            <div class="mt-2 text-[12px] leading-relaxed text-on-surface-variant">${escapeHtml(item.summary || item.description || "当前没有更多摘要。")}</div>
                            <div class="mt-3 text-[11px] leading-relaxed text-on-surface-variant">${escapeHtml(provenance.join(" · ") || "暂无溯源信息")}</div>
                        </div>
                        <span class="material-symbols-outlined text-on-surface-variant">unfold_more</span>
                    </div>
                </summary>
                <div class="mt-4 space-y-3 border-t border-outline-variant/10 pt-4">
                    <div class="text-[11px] text-on-surface-variant">建议技能名：${escapeHtml(item.suggested_skill_name || "未生成")}</div>
                    ${
                        skillPreview
                            ? `<pre class="overflow-auto rounded-xl border border-outline-variant/15 bg-black/20 p-3 text-[11px] leading-relaxed text-on-surface whitespace-pre-wrap">${escapeHtml(truncate(skillPreview, 1200))}</pre>`
                            : `<div class="text-[11px] text-on-surface-variant">当前没有可展示的技能草稿。</div>`
                    }
                </div>
            </details>
        `;
    }

    function renderMemoryView() {
        if (!$memoryView) {
            return;
        }

        if (memoryViewLoading) {
            $memoryView.innerHTML = `
                <div class="min-h-full px-6 py-8">
                    <div class="mx-auto max-w-6xl rounded-3xl border border-outline-variant/15 bg-surface-container-high/50 px-5 py-5">
                        <div class="flex items-center gap-3 text-on-surface-variant">
                            <div class="dots"><span></span><span></span><span></span></div>
                            <span class="text-sm">正在同步 Memory 页面数据…</span>
                        </div>
                    </div>
                </div>
            `;
            return;
        }

        const profileFields = userProfileSnapshot?.profile && typeof userProfileSnapshot.profile === "object"
            ? Object.entries(userProfileSnapshot.profile)
            : [];
        const profileMeta = userProfileSnapshot?.field_meta && typeof userProfileSnapshot.field_meta === "object"
            ? userProfileSnapshot.field_meta
            : {};
        const profileCards = profileFields.length
            ? profileFields.map(([fieldName, value]) => renderProfileFieldCard(fieldName, value, profileMeta[fieldName] || {})).join("")
            : renderLearningEmpty("当前账号还没有保存的用户画像字段。");
        const memoryCards = userMemoryEntries.length
            ? userMemoryEntries.map((entry) => renderMemoryEntryCard(entry)).join("")
            : renderLearningEmpty("当前账号还没有保存的长期记忆。");
        const workflowCards = learnedWorkflowCandidates.length
            ? learnedWorkflowCandidates.map((item) => renderWorkflowCandidateCard(item)).join("")
            : renderLearningEmpty("当前账号还没有学习到可审核的流程候选。");
        const errorBanner = memoryViewError
            ? `
                <div class="rounded-2xl border border-error/20 bg-error/10 px-4 py-3 text-sm text-error">
                    ${escapeHtml(memoryViewError)}
                </div>
            `
            : "";

        $memoryView.innerHTML = `
            <div class="min-h-full bg-surface">
                <div class="mx-auto w-full max-w-6xl space-y-6 px-6 py-8">
                    <section class="rounded-[32px] border border-primary/15 bg-[linear-gradient(145deg,rgba(0,238,252,0.10),rgba(17,20,23,0.97))] px-6 py-6 shadow-[0_18px_60px_rgba(0,238,252,0.08)]">
                        <div class="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
                            <div class="max-w-3xl">
                                <div class="text-[10px] font-mono uppercase tracking-[0.22em] text-primary/80">Account Memory Layer</div>
                                <h2 class="mt-2 text-2xl font-semibold text-on-surface">用户画像、长期记忆与 learned workflow 候选</h2>
                                <p class="mt-3 text-sm leading-relaxed text-on-surface-variant">
                                    这里统一管理当前账号范围内的长期信息。它们会跨会话复用，不再和当前运行的注入上下文混在同一个侧栏里。
                                </p>
                                <div class="mt-4 flex flex-wrap gap-2">
                                    <span class="inline-flex items-center rounded-full border border-primary/20 bg-primary/10 px-3 py-1 text-[10px] font-mono uppercase tracking-[0.18em] text-primary">账号级</span>
                                    <span class="inline-flex items-center rounded-full border border-secondary/20 bg-secondary/10 px-3 py-1 text-[10px] font-mono uppercase tracking-[0.18em] text-secondary">跨会话共享</span>
                                    <span class="inline-flex items-center rounded-full border border-outline-variant/20 bg-surface-container-high px-3 py-1 text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">支持修正与删除</span>
                                </div>
                            </div>
                            <div class="flex shrink-0 flex-col gap-3">
                                <button
                                    type="button"
                                    data-memory-action="refresh"
                                    class="inline-flex items-center justify-center gap-2 rounded-full border border-outline-variant/20 bg-surface-container-high px-3 py-2 text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant hover:text-primary hover:border-primary/40 transition-colors"
                                >
                                    <span class="material-symbols-outlined text-sm">refresh</span>
                                    <span>同步</span>
                                </button>
                                <div class="rounded-2xl border border-outline-variant/15 bg-black/10 px-4 py-3 text-[11px] leading-relaxed text-on-surface-variant">
                                    当前运行的注入检索上下文已移动到 Agent 页右侧的“学习”标签，仅在那里展示。
                                </div>
                            </div>
                        </div>
                        <div class="mt-6 grid gap-3 sm:grid-cols-3">
                            ${renderLearningMetricCard("画像字段", profileFields.length, "cyan")}
                            ${renderLearningMetricCard("长期记忆", userMemoryEntries.length, "violet")}
                            ${renderLearningMetricCard("流程候选", learnedWorkflowCandidates.length, "emerald")}
                        </div>
                    </section>

                    ${errorBanner}

                    <section class="grid gap-6 xl:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)]">
                        <div class="space-y-6">
                            <section class="space-y-4">
                                <div class="flex items-center justify-between gap-3">
                                    <div>
                                        <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">用户画像</div>
                                        <div class="mt-1 text-sm text-on-surface-variant">字段级画像会随账号长期保留，可直接修正或清除。</div>
                                    </div>
                                    <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-primary">${escapeHtml(String(profileFields.length))} 项</div>
                                </div>
                                <div class="space-y-3">${profileCards}</div>
                            </section>

                            <section class="space-y-4">
                                <div class="flex items-center justify-between gap-3">
                                    <div>
                                        <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">已保存记忆</div>
                                        <div class="mt-1 text-sm text-on-surface-variant">支持按关键词和类型筛选，并对错误记忆进行修正或删除。</div>
                                    </div>
                                    <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-secondary">${escapeHtml(String(userMemoryEntries.length))} 条</div>
                                </div>
                                <form data-memory-search-form class="rounded-2xl border border-outline-variant/15 bg-surface-container-high/60 px-4 py-4">
                                    <div class="grid gap-3">
                                        <input
                                            type="text"
                                            name="query"
                                            value="${escapeHtml(memoryQuery)}"
                                            placeholder="按摘要、内容或关键词筛选记忆"
                                            class="w-full rounded-xl border border-outline-variant/20 bg-surface-container-highest px-3 py-2 text-sm text-on-surface focus:border-primary/40 focus:ring-0"
                                        />
                                        <div class="grid gap-3 md:grid-cols-[minmax(0,1fr)_auto_auto]">
                                            <select
                                                name="memoryType"
                                                class="rounded-xl border border-outline-variant/20 bg-surface-container-highest px-3 py-2 text-sm text-on-surface focus:border-primary/40 focus:ring-0"
                                            >
                                                <option value="">全部类型</option>
                                                ${["preference", "communication_style", "goal", "constraint", "project_fact", "workflow_fact", "correction"]
                                                    .map((item) => `<option value="${item}" ${memoryTypeFilter === item ? "selected" : ""}>${formatMemoryTypeLabel(item)}</option>`)
                                                    .join("")}
                                            </select>
                                            <label class="inline-flex items-center gap-2 rounded-xl border border-outline-variant/15 bg-surface-container-highest px-3 py-2 text-[11px] text-on-surface-variant">
                                                <input type="checkbox" name="includeSuperseded" class="rounded border-outline-variant/30 bg-transparent text-primary focus:ring-0" ${memoryIncludeSuperseded ? "checked" : ""} />
                                                <span>含已替代</span>
                                            </label>
                                            <button
                                                type="submit"
                                                class="inline-flex items-center gap-1 rounded-xl border border-primary/30 bg-primary/10 px-3 py-2 text-[10px] font-mono uppercase tracking-[0.18em] text-primary hover:bg-primary/15 transition-colors"
                                            >
                                                <span class="material-symbols-outlined text-sm">search</span>
                                                <span>筛选</span>
                                            </button>
                                        </div>
                                    </div>
                                </form>
                                <div class="space-y-3">${memoryCards}</div>
                            </section>
                        </div>

                        <div class="space-y-6">
                            <section class="rounded-3xl border border-outline-variant/15 bg-[linear-gradient(145deg,rgba(148,163,184,0.08),rgba(15,23,42,0.92))] px-5 py-5">
                                <div class="text-[10px] font-mono uppercase tracking-[0.22em] text-primary/80">运行级信息已拆分</div>
                                <h3 class="mt-2 text-lg font-semibold text-on-surface">当前 run 的注入检索上下文</h3>
                                <p class="mt-2 text-sm leading-relaxed text-on-surface-variant">
                                    右侧“学习”标签现在只保留当前选中运行实际注入的 retrieved context，便于把账号级记忆和运行级检索来源分开看。
                                </p>
                            </section>

                            <section class="space-y-4">
                                <div class="flex items-center justify-between gap-3">
                                    <div>
                                        <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">learned workflow 候选</div>
                                        <div class="mt-1 text-sm text-on-surface-variant">这里展示从成功运行中提炼出的候选技能草稿和来源信号。</div>
                                    </div>
                                    <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-secondary">${escapeHtml(String(learnedWorkflowCandidates.length))} 条</div>
                                </div>
                                <div class="space-y-3">${workflowCards}</div>
                            </section>
                        </div>
                    </section>
                </div>
            </div>
        `;
    }

    function renderLearningPanel() {
        if (!$learningPanel) {
            return;
        }

        const retrievedSection = getRetrievedContextSection();
        const sourceCount = Array.isArray(retrievedSection?.sources) ? retrievedSection.sources.length : 0;
        const selectedRunLabel = selectedRunId ? selectedRunId.slice(0, 8) : "未选择";

        $learningPanel.innerHTML = `
            <section class="rounded-3xl border border-primary/15 bg-[linear-gradient(145deg,rgba(0,238,252,0.08),rgba(17,20,23,0.96))] px-5 py-5 shadow-[0_16px_50px_rgba(0,238,252,0.08)]">
                <div class="flex items-start justify-between gap-4">
                    <div>
                        <div class="text-[10px] font-mono uppercase tracking-[0.22em] text-primary/80">当前运行注入</div>
                        <h3 class="mt-2 text-lg font-semibold text-on-surface">检索上下文</h3>
                        <p class="mt-2 text-sm leading-relaxed text-on-surface-variant">
                            这里只展示当前选中运行实际注入的 retrieved context。账号级的用户画像、长期记忆和 learned workflow 候选已移到顶部导航里的 Memory 页面。
                        </p>
                    </div>
                    <span class="inline-flex items-center rounded-full border border-primary/20 bg-primary/10 px-3 py-1 text-[10px] font-mono uppercase tracking-[0.18em] text-primary">
                        Run ${escapeHtml(selectedRunLabel)}
                    </span>
                </div>
                <div class="mt-5 grid grid-cols-3 gap-3">
                    ${renderLearningMetricCard("条目", retrievedSection?.items || 0, "slate")}
                    ${renderLearningMetricCard("字符", retrievedSection?.chars || 0, "cyan")}
                    ${renderLearningMetricCard("来源", sourceCount, "violet")}
                </div>
            </section>

            <section class="space-y-4">
                <div class="flex items-center justify-between gap-3">
                    <div>
                        <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">retrieved_context</div>
                        <div class="mt-1 text-sm text-on-surface-variant">
                            ${
                                retrievedSection
                                    ? `当前 run 已注入“${escapeHtml(retrievedSection.title || "Recent Retrieved Context")}”区块，可直接检查正文和来源。`
                                    : "当前选中的运行还没有注入可展示的检索上下文。"
                            }
                        </div>
                    </div>
                    <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-primary">${escapeHtml(selectedRunLabel)}</div>
                </div>
                <div class="space-y-3">
                    ${retrievedSection ? renderPromptSectionCard(retrievedSection) : renderLearningEmpty("请选择一个已开始执行的运行，或等待当前运行生成检索注入信息。")}
                </div>
            </section>
        `;
    }

    async function refreshMemoryViewData() {
        if (!$memoryView) {
            return;
        }
        memoryViewLoading = true;
        memoryViewError = "";
        renderMemoryView();

        const params = new URLSearchParams({
            limit: "12",
        });
        if (memoryQuery.trim()) {
            params.set("query", memoryQuery.trim());
        }
        if (memoryTypeFilter) {
            params.set("memory_type", memoryTypeFilter);
        }
        if (memoryIncludeSuperseded) {
            params.set("include_superseded", "true");
        }

        try {
            const [profile, memories, workflows] = await Promise.all([
                fetchJson("/api/user-profile", { allow404: true }),
                fetchJson(`/api/user-memory?${params.toString()}`),
                fetchJson("/api/learned-workflows?limit=8").catch(() => []),
            ]);
            userProfileSnapshot = profile;
            userMemoryEntries = Array.isArray(memories) ? memories : [];
            learnedWorkflowCandidates = Array.isArray(workflows) ? workflows : [];
        } catch (error) {
            memoryViewError = `同步失败：${error.message || "未知错误"}`;
        } finally {
            memoryViewLoading = false;
            renderMemoryView();
        }
    }

    function closeLearningEditorModal() {
        const modal = document.getElementById("learning-editor-modal");
        if (!modal) {
            return;
        }
        modal.remove();
    }

    function openLearningEditorModal({ title, subtitle = "", bodyHtml = "", onSubmit }) {
        closeLearningEditorModal();
        const modal = document.createElement("div");
        modal.id = "learning-editor-modal";
        modal.className = "fixed inset-0 z-[220] flex items-center justify-center bg-black/70 px-4 backdrop-blur-sm";
        modal.innerHTML = `
            <div class="w-full max-w-2xl rounded-[28px] border border-outline-variant/15 bg-surface shadow-[0_24px_80px_rgba(0,0,0,0.45)]">
                <div class="flex items-start justify-between gap-4 border-b border-outline-variant/10 px-5 py-4">
                    <div class="min-w-0">
                        <div class="text-[10px] font-mono uppercase tracking-[0.22em] text-primary/80">记忆层修正</div>
                        <h3 class="mt-2 text-lg font-semibold text-on-surface">${escapeHtml(title)}</h3>
                        ${subtitle ? `<p class="mt-2 text-sm leading-relaxed text-on-surface-variant">${escapeHtml(subtitle)}</p>` : ""}
                    </div>
                    <button
                        type="button"
                        data-learning-modal-action="close"
                        class="inline-flex h-10 w-10 items-center justify-center rounded-full border border-outline-variant/20 bg-surface-container-highest text-on-surface-variant transition-colors hover:border-primary/40 hover:text-on-surface"
                        title="关闭"
                    >
                        <span class="material-symbols-outlined text-sm">close</span>
                    </button>
                </div>
                <form id="learning-editor-form" class="space-y-4 px-5 py-5">
                    ${bodyHtml}
                    <div id="learning-editor-error" class="hidden rounded-2xl border border-error/20 bg-error/10 px-4 py-3 text-sm text-error"></div>
                    <div class="flex items-center justify-end gap-3 border-t border-outline-variant/10 pt-4">
                        <button
                            type="button"
                            data-learning-modal-action="close"
                            class="inline-flex items-center gap-2 rounded-xl border border-outline-variant/20 bg-surface-container-high px-4 py-2 text-sm text-on-surface-variant hover:border-outline-variant/40 hover:text-on-surface transition-colors"
                        >
                            <span>取消</span>
                        </button>
                        <button
                            type="submit"
                            class="inline-flex items-center gap-2 rounded-xl border border-primary/30 bg-primary/10 px-4 py-2 text-sm font-semibold text-primary hover:bg-primary/15 transition-colors"
                        >
                            <span class="material-symbols-outlined text-sm">save</span>
                            <span>保存修正</span>
                        </button>
                    </div>
                </form>
            </div>
        `;
        document.body.appendChild(modal);

        const form = modal.querySelector("#learning-editor-form");
        const errorEl = modal.querySelector("#learning-editor-error");
        const close = () => closeLearningEditorModal();

        modal.addEventListener("click", (event) => {
            if (event.target === modal || event.target.closest("[data-learning-modal-action='close']")) {
                close();
            }
        });
        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            if (!onSubmit) {
                close();
                return;
            }
            errorEl.classList.add("hidden");
            errorEl.textContent = "";
            try {
                await onSubmit(new FormData(form));
                close();
            } catch (error) {
                errorEl.textContent = error.message || "保存失败，请稍后重试。";
                errorEl.classList.remove("hidden");
            }
        });
    }

    function parseProfileFieldInput(fieldName, rawValue) {
        if (["recurring_projects", "dislikes_avoidances"].includes(fieldName)) {
            return String(rawValue || "")
                .split(/[\n,，]/)
                .map((item) => item.trim())
                .filter(Boolean);
        }
        return String(rawValue || "").trim();
    }

    function openProfileFieldEditor(fieldName) {
        const currentValue = userProfileSnapshot?.profile?.[fieldName];
        const isListField = Array.isArray(currentValue) || ["recurring_projects", "dislikes_avoidances"].includes(fieldName);
        openLearningEditorModal({
            title: `修正画像字段：${formatProfileFieldLabel(fieldName)}`,
            subtitle: isListField ? "多项内容可用换行或逗号分隔。" : "请直接输入新的字段值。",
            bodyHtml: `
                <label class="block space-y-2">
                    <span class="text-[11px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">字段值</span>
                    <textarea
                        name="value"
                        rows="${isListField ? 5 : 4}"
                        class="w-full rounded-2xl border border-outline-variant/20 bg-surface-container-highest px-4 py-3 text-sm text-on-surface focus:border-primary/40 focus:ring-0"
                    >${escapeHtml(Array.isArray(currentValue) ? currentValue.join("\n") : String(currentValue ?? ""))}</textarea>
                </label>
            `,
            onSubmit: async (formData) => {
                const nextValue = parseProfileFieldInput(fieldName, formData.get("value"));
                if (!Array.isArray(nextValue) && !String(nextValue || "").trim()) {
                    throw new Error("字段值不能为空。");
                }
                await fetchJson("/api/user-profile", {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        profile_updates: {
                            [fieldName]: nextValue,
                        },
                        profile_source: "explicit",
                        profile_confidence: 1.0,
                    }),
                });
                setStatus(`已更新画像字段：${formatProfileFieldLabel(fieldName)}`);
                await refreshMemoryViewData();
            },
        });
    }

    async function removeProfileField(fieldName) {
        const confirmed = window.confirm(`确认清除画像字段“${formatProfileFieldLabel(fieldName)}”吗？`);
        if (!confirmed) {
            return;
        }
        await fetchJson("/api/user-profile", {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                remove_fields: [fieldName],
                profile_source: "explicit",
                profile_confidence: 1.0,
            }),
        });
        setStatus(`已清除画像字段：${formatProfileFieldLabel(fieldName)}`);
        await refreshMemoryViewData();
    }

    function openMemoryEntryEditor(entryId) {
        const entry = userMemoryEntries.find((item) => item.id === entryId);
        if (!entry) {
            return;
        }
        openLearningEditorModal({
            title: `修正长期记忆：${formatMemoryTypeLabel(entry.memory_type)}`,
            subtitle: "这会直接更新当前记忆条目，并记录新的写入来源。",
            bodyHtml: `
                <div class="grid gap-4">
                    <label class="block space-y-2">
                        <span class="text-[11px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">记忆类型</span>
                        <select
                            name="memory_type"
                            class="w-full rounded-2xl border border-outline-variant/20 bg-surface-container-highest px-4 py-3 text-sm text-on-surface focus:border-primary/40 focus:ring-0"
                        >
                            ${["preference", "communication_style", "goal", "constraint", "project_fact", "workflow_fact", "correction"]
                                .map((item) => `<option value="${item}" ${entry.memory_type === item ? "selected" : ""}>${formatMemoryTypeLabel(item)}</option>`)
                                .join("")}
                        </select>
                    </label>
                    <label class="block space-y-2">
                        <span class="text-[11px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">摘要</span>
                        <textarea
                            name="summary"
                            rows="3"
                            class="w-full rounded-2xl border border-outline-variant/20 bg-surface-container-highest px-4 py-3 text-sm text-on-surface focus:border-primary/40 focus:ring-0"
                        >${escapeHtml(entry.summary || "")}</textarea>
                    </label>
                    <label class="block space-y-2">
                        <span class="text-[11px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">正文</span>
                        <textarea
                            name="content"
                            rows="7"
                            class="w-full rounded-2xl border border-outline-variant/20 bg-surface-container-highest px-4 py-3 text-sm text-on-surface focus:border-primary/40 focus:ring-0"
                        >${escapeHtml(entry.content || "")}</textarea>
                    </label>
                    <label class="block space-y-2">
                        <span class="text-[11px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">置信度</span>
                        <input
                            type="number"
                            name="confidence"
                            min="0"
                            max="1"
                            step="0.01"
                            value="${escapeHtml(String(entry.confidence ?? 1))}"
                            class="w-full rounded-2xl border border-outline-variant/20 bg-surface-container-highest px-4 py-3 text-sm text-on-surface focus:border-primary/40 focus:ring-0"
                        />
                    </label>
                </div>
            `,
            onSubmit: async (formData) => {
                const content = String(formData.get("content") || "").trim();
                if (!content) {
                    throw new Error("记忆正文不能为空。");
                }
                const confidence = Number(formData.get("confidence"));
                if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
                    throw new Error("置信度必须在 0 到 1 之间。");
                }
                await fetchJson(`/api/user-memory/${encodeURIComponent(entryId)}`, {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        memory_type: String(formData.get("memory_type") || entry.memory_type),
                        summary: String(formData.get("summary") || "").trim(),
                        content,
                        confidence,
                    }),
                });
                setStatus("已更新长期记忆。");
                await refreshMemoryViewData();
            },
        });
    }

    async function deleteMemoryEntry(entryId) {
        const entry = userMemoryEntries.find((item) => item.id === entryId);
        const summary = entry?.summary || entry?.content || entryId;
        const confirmed = window.confirm(`确认删除这条长期记忆吗？\n\n${truncate(summary, 120)}`);
        if (!confirmed) {
            return;
        }
        await fetchJson(`/api/user-memory/${encodeURIComponent(entryId)}`, {
            method: "DELETE",
        });
        setStatus("已删除长期记忆。");
        await refreshMemoryViewData();
    }

    async function handleMemoryViewSubmit(event) {
        const form = event.target.closest("[data-memory-search-form]");
        if (!form) {
            return;
        }
        event.preventDefault();
        memoryQuery = String(form.querySelector("[name='query']")?.value || "").trim();
        memoryTypeFilter = String(form.querySelector("[name='memoryType']")?.value || "").trim();
        memoryIncludeSuperseded = Boolean(form.querySelector("[name='includeSuperseded']")?.checked);
        await refreshMemoryViewData();
    }

    async function handleMemoryViewClick(event) {
        const actionButton = event.target.closest("[data-memory-action]");
        if (!actionButton) {
            return;
        }
        const action = actionButton.dataset.memoryAction;
        if (action === "refresh") {
            await refreshMemoryViewData();
            return;
        }
        if (action === "edit-profile-field") {
            openProfileFieldEditor(actionButton.dataset.fieldName || "");
            return;
        }
        if (action === "remove-profile-field") {
            await removeProfileField(actionButton.dataset.fieldName || "");
            return;
        }
        if (action === "edit-memory-entry") {
            openMemoryEntryEditor(actionButton.dataset.entryId || "");
            return;
        }
        if (action === "delete-memory-entry") {
            await deleteMemoryEntry(actionButton.dataset.entryId || "");
        }
    }

    function renderRunBanner() {
        if (!$activeRunBanner || !$activeRunBannerStatus || !$activeRunBannerGoal || !$activeRunBannerMeta) {
            return;
        }
        if (!isFeatureEnabled("enable_durable_runs")) {
            $activeRunBanner.classList.add("hidden");
            $activeRunBanner.classList.remove("flex");
            if ($headerOpenRunsBtn) {
                $headerOpenRunsBtn.classList.add("hidden");
                $headerOpenRunsBtn.classList.remove("inline-flex");
            }
            return;
        }

        if ($headerOpenRunsBtn) {
            const showInspectorEntry = Boolean(sessionId);
            $headerOpenRunsBtn.classList.toggle("hidden", !showInspectorEntry);
            $headerOpenRunsBtn.classList.toggle("inline-flex", showInspectorEntry);
        }

        const activeRun = getCurrentRootRun();

        if (!activeRun) {
            $activeRunBanner.classList.add("hidden");
            $activeRunBanner.classList.remove("flex");
            return;
        }

        const meta = getRunStatusMeta(activeRun.status);
        $activeRunBanner.classList.remove("hidden");
        $activeRunBanner.classList.add("flex");
        $activeRunBannerStatus.className = `inline-flex items-center gap-1 rounded-full border px-2 py-1 ${meta.tone}`;
        $activeRunBannerStatus.innerHTML = `
            <span class="material-symbols-outlined text-sm">${meta.icon}</span>
            <span>${escapeHtml(meta.label)}</span>
        `;
        $activeRunBannerGoal.textContent = activeRun.goal || "当前任务";
        $activeRunBannerMeta.textContent = buildRunMetaLine(activeRun);
    }

    function renderRunSessionSummary() {
        if (!$runSessionSummary) return;
        if (!isFeatureEnabled("enable_durable_runs")) {
            $runSessionSummary.innerHTML = "";
            return;
        }
        const snapshot = buildSessionStatusSnapshot();

        $runSessionSummary.innerHTML = `
            <article class="rounded-3xl border border-outline-variant/10 bg-gradient-to-br from-surface-container-high via-surface-container-high to-surface-container-low px-4 py-4">
                <div class="flex items-start justify-between gap-3">
                    <div class="min-w-0">
                        <div class="text-[10px] font-mono uppercase tracking-[0.22em] text-on-surface-variant">${escapeHtml(snapshot.eyebrow)}</div>
                        <h3 class="mt-2 text-sm font-semibold leading-relaxed text-on-surface">${escapeHtml(snapshot.title)}</h3>
                        <p class="mt-2 text-[11px] leading-relaxed text-on-surface-variant">${escapeHtml(snapshot.body)}</p>
                    </div>
                    <span class="inline-flex items-center rounded-full border px-2.5 py-1 text-[10px] font-mono uppercase tracking-[0.18em] ${snapshot.badgeTone}">
                        ${escapeHtml(snapshot.badgeLabel)}
                    </span>
                </div>
            </article>
        `;
    }

    function renderApprovalInbox() {
        if (!$approvalInboxList || !$approvalInboxCount || !$approvalInboxShell || !$primaryAttentionPanel) return;
        if (!isFeatureEnabled("enable_approval_flow")) {
            $approvalInboxCount.textContent = "0";
            $approvalInboxList.innerHTML = "";
            $approvalInboxShell.classList.add("hidden");
            $primaryAttentionPanel.classList.add("hidden");
            return;
        }
        $approvalInboxCount.textContent = String(sessionPendingApprovals.length);

        if (!sessionPendingApprovals.length) {
            $approvalInboxList.innerHTML = "";
            $approvalInboxShell.classList.add("hidden");
            $primaryAttentionPanel.classList.add("hidden");
            return;
        }

        $primaryAttentionPanel.classList.remove("hidden");
        $approvalInboxShell.classList.remove("hidden");
        $approvalInboxList.innerHTML = sessionPendingApprovals
            .map((approval) => renderApprovalRecord(approval, { compact: true }))
            .join("");
    }

    function renderRunActiveCard() {
        if (!$runActiveCard) return;
        const activeRun = getCurrentRootRun();
        const focusRun = activeRun || getSelectedRun() || getRootRuns()[0] || null;
        if (!focusRun) {
            $runActiveCard.innerHTML = `
                <div class="rounded-2xl border border-dashed border-outline-variant/20 bg-surface-container-low px-4 py-5">
                    <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">当前任务</div>
                    <div class="mt-3 text-sm text-on-surface-variant">当前还没有正在执行的任务，可以从下方输入框开始。</div>
                </div>
            `;
            return;
        }

        const meta = getRunStatusMeta(focusRun.status);
        const isLive = Boolean(activeRun && activeRun.id === focusRun.id);
        const pendingCount = sessionPendingApprovals.filter((approval) => approval.run_id === focusRun.id).length;
        $runActiveCard.innerHTML = `
            <article class="rounded-2xl border border-primary/20 bg-gradient-to-br from-primary/10 via-surface-container-high to-surface-container-highest px-4 py-4 shadow-[0_0_30px_rgba(0,240,255,0.08)]">
                <div class="flex items-start justify-between gap-3">
                    <div class="min-w-0">
                        <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-primary">${escapeHtml(isLive ? "当前任务" : "最新任务")}</div>
                        <h3 class="mt-2 text-sm font-bold text-on-surface leading-relaxed">${escapeHtml(focusRun.goal || "未命名任务")}</h3>
                        <p class="mt-2 text-[11px] text-on-surface-variant">${escapeHtml(buildRunMetaLine(focusRun))}</p>
                    </div>
                    <span class="inline-flex items-center gap-1 rounded-full border px-2 py-1 text-[10px] font-mono uppercase tracking-[0.18em] ${meta.tone}">
                        <span class="material-symbols-outlined text-sm">${meta.icon}</span>
                        <span>${escapeHtml(meta.label)}</span>
                    </span>
                </div>
                <div class="mt-4 space-y-3">
                    <div class="rounded-xl border border-outline-variant/10 bg-black/20 px-3 py-3">
                        <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">当前进展</div>
                        <div class="mt-2 text-[12px] leading-relaxed text-on-surface">${escapeHtml(buildRunFocusSummary(focusRun, { isLive, pendingCount }))}</div>
                    </div>
                    <div class="rounded-xl border border-outline-variant/10 bg-black/20 px-3 py-3">
                        <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">下一步</div>
                        <div class="mt-2 text-[12px] leading-relaxed text-on-surface">${escapeHtml(buildRunNextAction(focusRun, { pendingCount }))}</div>
                    </div>
                </div>
                <div class="mt-4 flex flex-wrap gap-2">
                    <button data-run-action="select" data-run-id="${escapeHtml(focusRun.id)}" class="inline-flex items-center gap-2 rounded-xl border border-primary/30 bg-primary/10 px-3 py-2 text-[10px] font-mono uppercase tracking-[0.18em] text-primary hover:bg-primary/15 transition-colors">
                        <span class="material-symbols-outlined text-sm">pageview</span>
                        <span>打开详情</span>
                    </button>
                    ${
                        isRunRecoverable(focusRun)
                            ? `<button data-run-action="resume" data-run-id="${escapeHtml(focusRun.id)}" class="inline-flex items-center gap-2 rounded-xl border border-primary/30 bg-primary/10 px-3 py-2 text-[10px] font-mono uppercase tracking-[0.18em] text-primary hover:bg-primary/15 transition-colors">
                                <span class="material-symbols-outlined text-sm">restart_alt</span>
                                <span>继续执行</span>
                               </button>`
                            : ""
                    }
                    ${
                        isRunActive(focusRun)
                            ? `<button data-run-action="cancel" data-run-id="${escapeHtml(focusRun.id)}" class="inline-flex items-center gap-2 rounded-xl border border-error/20 bg-error/10 px-3 py-2 text-[10px] font-mono uppercase tracking-[0.18em] text-error hover:bg-error/15 transition-colors">
                                <span class="material-symbols-outlined text-sm">stop_circle</span>
                                <span>取消执行</span>
                               </button>`
                            : ""
                    }
                </div>
            </article>
        `;
    }

    function renderRunHistory() {
        if (!$runHistoryList || !$runHistoryCount) return;
        const rootRuns = getRootRuns();
        $runHistoryCount.textContent = String(rootRuns.length);
        if (!rootRuns.length) {
            $runHistoryList.innerHTML = `
                <div class="rounded-2xl border border-dashed border-outline-variant/20 bg-surface-container-low px-4 py-5 text-sm text-on-surface-variant">
                    当前会话还没有执行记录。
                </div>
            `;
            return;
        }

        $runHistoryList.innerHTML = rootRuns
            .slice(0, 3)
            .map((run) => {
                const meta = getRunStatusMeta(run.status);
                const isSelected = run.id === selectedRunId;
                return `
                    <button
                        type="button"
                        data-run-action="select"
                        data-run-id="${escapeHtml(run.id)}"
                        class="w-full rounded-2xl border px-4 py-4 text-left transition-all ${
                            isSelected
                                ? "border-secondary/40 bg-secondary/10 shadow-[inset_0_0_0_1px_rgba(213,117,255,0.2)]"
                                : "border-outline-variant/10 bg-surface-container-high hover:border-primary/30 hover:bg-surface-container-highest"
                        }"
                    >
                        <div class="flex items-start justify-between gap-3">
                            <div class="min-w-0">
                                <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">${escapeHtml(isSelected ? "已选任务" : "最近任务")}</div>
                                <div class="mt-2 text-sm font-bold text-on-surface leading-relaxed">${escapeHtml(run.goal || "未命名任务")}</div>
                            </div>
                            <span class="inline-flex items-center gap-1 rounded-full border px-2 py-1 text-[10px] font-mono uppercase tracking-[0.18em] ${meta.tone}">
                                <span class="material-symbols-outlined text-sm">${meta.icon}</span>
                                <span>${escapeHtml(meta.label)}</span>
                            </span>
                        </div>
                        <div class="mt-3 text-[11px] leading-relaxed text-on-surface-variant">
                            ${escapeHtml(buildRunMetaLine(run))}
                        </div>
                        ${
                            run.error_summary
                                ? `<div class="mt-3 rounded-xl border border-error/20 bg-error/10 px-3 py-2 text-[11px] text-error/90">${escapeHtml(truncate(run.error_summary, 220))}</div>`
                                : ""
                        }
                    </button>
                `;
            })
            .join("");
    }

    function getArtifactRolePresentation(artifact) {
        const role = String(artifact?.role || "").trim();
        const source = String(artifact?.source || "").trim();
        const isFinal = Boolean(artifact?.is_final);

        if (isFinal && role === "revised_file") {
            return {
                label: "修订交付",
                tone: "text-emerald-400 border-emerald-400/30 bg-emerald-400/10",
            };
        }
        if (isFinal || role === "final_deliverable") {
            return {
                label: "最终交付",
                tone: "text-emerald-400 border-emerald-400/30 bg-emerald-400/10",
            };
        }
        if (role === "revised_file" || source === "agent_revised") {
            return {
                label: "修订文件",
                tone: "text-secondary border-secondary/30 bg-secondary/10",
            };
        }
        if (role === "input_source" || source === "user_uploaded") {
            return {
                label: "源文件",
                tone: "text-sky-300 border-sky-300/30 bg-sky-300/10",
            };
        }
        return {
            label: "运行产出",
            tone: "text-primary border-primary/20 bg-primary/10",
        };
    }

    function formatArtifactTypeLabel(artifactType) {
        const normalized = String(artifactType || "").trim().toLowerCase();
        const labels = {
            document: "文档产出",
            workspace_file: "工作区文件",
            background_process: "后台进程",
        };
        return labels[normalized] || artifactType || "产出";
    }

    function describeArtifactLineage(artifact) {
        const metadata = artifact?.metadata && typeof artifact.metadata === "object"
            ? artifact.metadata
            : {};
        const uploadName = String(
            metadata.parent_upload_name
            || metadata.parent_upload_relative_path
            || metadata.parent_upload_id
            || ""
        ).trim();
        if (!uploadName) {
            return "";
        }
        const revisionMode = metadata.revision_mode === "overwrite"
            ? "直接覆盖了上传原件"
            : "基于上传文件生成了副本";
        return `${revisionMode}：${uploadName}`;
    }

    function findPrimaryDeliverableArtifact(run) {
        const manifest = run?.deliverable_manifest && typeof run.deliverable_manifest === "object"
            ? run.deliverable_manifest
            : { primary_artifact_id: null, items: [] };
        const artifactMap = new Map(
            selectedRunArtifacts.map((artifact) => [artifact.id, artifact])
        );
        if (manifest.primary_artifact_id && artifactMap.has(manifest.primary_artifact_id)) {
            return artifactMap.get(manifest.primary_artifact_id) || null;
        }
        if (Array.isArray(manifest.items)) {
            for (const item of manifest.items) {
                if (item?.artifact_id && artifactMap.has(item.artifact_id)) {
                    return artifactMap.get(item.artifact_id) || null;
                }
            }
        }
        return selectedRunArtifacts.find((artifact) => artifact.is_final || artifact.role === "final_deliverable") || null;
    }

    function renderArtifactCard(artifact, { headingLabel = "", emphasize = false } = {}) {
        if (!artifact) {
            return "";
        }
        const badge = getArtifactRolePresentation(artifact);
        const lineage = describeArtifactLineage(artifact);
        const displayName = artifact.display_name || artifact.uri || artifact.id || "未命名文件";
        const formatLabel = String(artifact.format || artifact.preview_kind || "").trim();
        const summary = String(artifact.summary || "").trim();
        const uriLabel = String(artifact.uri || "").trim();
        const fileDescriptor = createArtifactFileDescriptor(artifact);

        return `
            <div class="mb-2 rounded-xl border ${emphasize ? "border-emerald-400/25 bg-emerald-400/5" : "border-outline-variant/10 bg-surface-container-high"} px-3 py-3">
                <div class="flex items-start justify-between gap-3">
                    <div class="min-w-0">
                        ${headingLabel ? `<div class="text-[10px] font-mono uppercase tracking-[0.18em] ${emphasize ? "text-emerald-400" : "text-on-surface-variant"}">${escapeHtml(headingLabel)}</div>` : ""}
                        <div class="mt-1 text-sm text-on-surface break-all">${escapeHtml(displayName)}</div>
                    </div>
                    <div class="flex flex-wrap justify-end gap-2">
                        <span class="inline-flex items-center rounded-full border px-2 py-1 text-[10px] font-mono uppercase tracking-[0.18em] ${badge.tone}">
                            ${escapeHtml(badge.label)}
                        </span>
                        ${formatLabel ? `
                            <span class="inline-flex items-center rounded-full border border-outline-variant/20 bg-black/20 px-2 py-1 text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">
                                ${escapeHtml(formatLabel)}
                            </span>
                        ` : ""}
                    </div>
                </div>
                ${uriLabel ? `<div class="mt-2 text-[11px] text-on-surface-variant break-all">${escapeHtml(uriLabel)}</div>` : ""}
                ${lineage ? `<div class="mt-2 text-[11px] text-secondary">${escapeHtml(lineage)}</div>` : ""}
                ${summary ? `<div class="mt-2 text-[11px] text-on-surface-variant">${escapeHtml(truncate(summary, 180))}</div>` : ""}
                ${fileDescriptor ? `<div class="mt-4">${renderFileActionButtons(fileDescriptor)}</div>` : ""}
            </div>
        `;
    }

    function getArtifactIdentityKey(artifact) {
        return String(artifact?.id || artifact?.uri || artifact?.display_name || "").trim();
    }

    function normalizeArtifactForConversationShelf(artifact) {
        if (!artifact || typeof artifact !== "object") {
            return null;
        }
        const identityKey = getArtifactIdentityKey(artifact);
        if (!identityKey) {
            return null;
        }
        return {
            ...artifact,
            id: artifact.id || "",
            artifact_type: artifact.artifact_type || "workspace_file",
            display_name: artifact.display_name || artifact.uri || artifact.id || "未命名文件",
            role: artifact.role || "",
            format: artifact.format || "",
            mime_type: artifact.mime_type || "",
            preview_kind: artifact.preview_kind || guessPreviewKindFromFileMeta(
                artifact.display_name || artifact.uri || artifact.id || "",
                artifact.mime_type || ""
            ),
            summary: artifact.summary || "",
            metadata: artifact.metadata && typeof artifact.metadata === "object" ? artifact.metadata : {},
            is_final: Boolean(artifact.is_final),
        };
    }

    function getAssistantArtifactShelfState(container) {
        let state = assistantArtifactShelves.get(container);
        if (!state) {
            state = {
                artifacts: new Map(),
                primaryArtifactId: "",
            };
            assistantArtifactShelves.set(container, state);
        }
        return state;
    }

    function removeAssistantArtifactShelf(container) {
        if (!container) {
            return;
        }
        const existingShelf = container.querySelector(".assistant-artifact-shelf");
        if (existingShelf) {
            existingShelf.remove();
        }
        const state = getAssistantArtifactShelfState(container);
        state.artifacts.clear();
        state.primaryArtifactId = "";
    }

    function renderAssistantArtifactShelf(container) {
        if (!container) {
            return;
        }
        const state = getAssistantArtifactShelfState(container);
        const artifacts = Array.from(state.artifacts.values());
        const existingShelf = container.querySelector(".assistant-artifact-shelf");
        if (!artifacts.length) {
            if (existingShelf) {
                existingShelf.remove();
            }
            return;
        }

        const primaryArtifact = state.primaryArtifactId
            ? artifacts.find((artifact) => artifact.id === state.primaryArtifactId) || null
            : artifacts.find((artifact) => artifact.is_final || artifact.role === "final_deliverable" || artifact.role === "revised_file") || null;
        const primaryKey = primaryArtifact ? getArtifactIdentityKey(primaryArtifact) : "";
        const supportingArtifacts = artifacts.filter((artifact) => getArtifactIdentityKey(artifact) !== primaryKey);
        const linkedArtifactCount = artifacts.filter((artifact) => artifact.id).length;
        const cards = [];

        if (primaryArtifact) {
            cards.push(
                renderArtifactCard(primaryArtifact, {
                    headingLabel: "本轮交付",
                    emphasize: true,
                })
            );
        }

        supportingArtifacts.slice(0, 3).forEach((artifact) => {
            cards.push(
                renderArtifactCard(artifact, {
                    headingLabel: "补充文件",
                })
            );
        });

        const shelf = existingShelf || document.createElement("section");
        shelf.className = "assistant-artifact-shelf ml-4 mb-2 rounded-2xl border border-secondary/20 bg-secondary/5 px-4 py-4";
        shelf.innerHTML = `
            <div class="flex items-start justify-between gap-3">
                <div class="min-w-0">
                    <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-secondary">交付结果</div>
                    <div class="mt-2 text-sm leading-relaxed text-on-surface">
                        ${escapeHtml(
                            linkedArtifactCount === artifacts.length
                                ? "本轮产出已经整理为可查看文件，可以直接在对话里预览、打开或下载。"
                                : "本轮产出已经整理到对话流里；正式预览链接会在运行记录同步后自动补齐。"
                        )}
                    </div>
                </div>
                <span class="inline-flex items-center rounded-full border border-secondary/20 bg-black/20 px-2 py-1 text-[10px] font-mono uppercase tracking-[0.18em] text-secondary">
                    ${escapeHtml(`${artifacts.length} 个文件`)}
                </span>
            </div>
            <div class="mt-4">
                ${cards.join("")}
            </div>
        `;

        if (!existingShelf) {
            insertBeforeDots(container, shelf);
        }
    }

    function upsertAssistantArtifactShelf(
        container,
        artifacts,
        { replace = false, primaryArtifactId = "" } = {}
    ) {
        if (!container) {
            return;
        }
        const state = getAssistantArtifactShelfState(container);
        if (replace) {
            state.artifacts.clear();
            state.primaryArtifactId = "";
        }
        const normalizedArtifacts = Array.isArray(artifacts)
            ? artifacts
                .map((artifact) => normalizeArtifactForConversationShelf(artifact))
                .filter(Boolean)
            : [];

        if (replace && !normalizedArtifacts.length) {
            removeAssistantArtifactShelf(container);
            return;
        }

        normalizedArtifacts.forEach((artifact) => {
            state.artifacts.set(getArtifactIdentityKey(artifact), artifact);
        });
        if (primaryArtifactId) {
            state.primaryArtifactId = primaryArtifactId;
        }
        renderAssistantArtifactShelf(container);
    }

    function syncAssistantArtifactShelfWithRun(container, run, artifacts) {
        const manifest = run?.deliverable_manifest && typeof run.deliverable_manifest === "object"
            ? run.deliverable_manifest
            : { primary_artifact_id: "" };
        upsertAssistantArtifactShelf(container, artifacts, {
            replace: true,
            primaryArtifactId: manifest.primary_artifact_id || "",
        });
    }

    function findLatestAssistantStream() {
        const streams = Array.from($messages.querySelectorAll(".message.assistant .stream"));
        return streams.length ? streams[streams.length - 1] : null;
    }

    function resetConversationAssistantTurns() {
        conversationAssistantTurns = [];
    }

    function resetConversationArtifactCache() {
        conversationArtifactsByRunId = new Map();
        conversationArtifactFetchKeys = new Map();
    }

    function getCachedConversationArtifacts(runId) {
        const rootRunId = resolveRootRunId(runId);
        if (!rootRunId) {
            return [];
        }
        const cachedArtifacts = conversationArtifactsByRunId.get(rootRunId);
        return Array.isArray(cachedArtifacts) ? cachedArtifacts : [];
    }

    function mergeConversationArtifactsForRun(runId, artifacts) {
        const rootRunId = resolveRootRunId(runId);
        if (!rootRunId) {
            return;
        }
        const nextArtifacts = Array.isArray(artifacts)
            ? artifacts
                .map((artifact) => normalizeArtifactForConversationShelf(artifact))
                .filter(Boolean)
            : [];
        if (!nextArtifacts.length) {
            return;
        }
        const mergedArtifacts = new Map();
        getCachedConversationArtifacts(rootRunId).forEach((artifact) => {
            mergedArtifacts.set(getArtifactIdentityKey(artifact), artifact);
        });
        nextArtifacts.forEach((artifact) => {
            mergedArtifacts.set(getArtifactIdentityKey(artifact), artifact);
        });
        conversationArtifactsByRunId.set(rootRunId, Array.from(mergedArtifacts.values()));
    }

    function replaceConversationArtifactsForRun(runId, artifacts, { updateFetchKey = false } = {}) {
        const rootRunId = resolveRootRunId(runId);
        if (!rootRunId) {
            return;
        }
        const nextArtifacts = Array.isArray(artifacts)
            ? artifacts
                .map((artifact) => normalizeArtifactForConversationShelf(artifact))
                .filter(Boolean)
            : [];
        conversationArtifactsByRunId.set(rootRunId, nextArtifacts);
        if (!updateFetchKey) {
            return;
        }
        const rootRun = getRunById(rootRunId);
        if (rootRun) {
            conversationArtifactFetchKeys.set(rootRunId, buildConversationArtifactFetchKey(rootRun));
        }
    }

    function compareTimestampAsc(leftValue, rightValue) {
        const leftTime = Date.parse(leftValue || "") || 0;
        const rightTime = Date.parse(rightValue || "") || 0;
        return leftTime - rightTime;
    }

    function getRootRunsChronologically() {
        return sessionRuns
            .filter((run) => !run.parent_run_id)
            .slice()
            .sort((left, right) => compareTimestampAsc(left.created_at, right.created_at));
    }

    function buildConversationArtifactsFromRunManifest(run) {
        const manifest = run?.deliverable_manifest && typeof run.deliverable_manifest === "object"
            ? run.deliverable_manifest
            : { items: [] };
        const items = Array.isArray(manifest.items) ? manifest.items : [];
        return items.map((item) => ({
            id: item.artifact_id || "",
            artifact_type: "document",
            uri: item.uri || "",
            display_name: item.display_name || item.uri || item.artifact_id || "未命名文件",
            role: item.role || "final_deliverable",
            format: item.format || "",
            mime_type: item.mime_type || "",
            preview_kind: guessPreviewKindFromFileMeta(
                item.display_name || item.uri || item.artifact_id || "",
                item.mime_type || ""
            ),
            is_final: true,
            summary: "",
            metadata: {},
        }));
    }

    function syncConversationArtifactShelvesWithRuns() {
        if (!conversationAssistantTurns.length) {
            return;
        }

        const rootRuns = getRootRunsChronologically();
        const remainingRootRuns = rootRuns.slice();
        conversationAssistantTurns.forEach((container) => {
            const explicitRootRunId = getAssistantStreamRootRunId(container);
            let run = explicitRootRunId
                ? remainingRootRuns.find((item) => item.id === explicitRootRunId) || null
                : null;
            if (run) {
                const matchedIndex = remainingRootRuns.findIndex((item) => item.id === run.id);
                if (matchedIndex >= 0) {
                    remainingRootRuns.splice(matchedIndex, 1);
                }
            } else {
                run = remainingRootRuns.shift() || null;
                if (run) {
                    setAssistantStreamRunId(container, run.id);
                }
            }
            if (!run) {
                removeAssistantArtifactShelf(container);
                return;
            }

            const cachedArtifacts = getCachedConversationArtifacts(run.id);
            const artifacts = Array.isArray(cachedArtifacts) && cachedArtifacts.length
                ? cachedArtifacts
                : run.id === selectedRunId && selectedRunArtifacts.length
                ? selectedRunArtifacts
                : buildConversationArtifactsFromRunManifest(run);

            if (artifacts.length) {
                syncAssistantArtifactShelfWithRun(container, run, artifacts);
                return;
            }

            if (RUN_ACTIVE_STATUSES.has(String(run.status || "").trim())) {
                return;
            }
            removeAssistantArtifactShelf(container);
        });
    }

    function buildConversationArtifactFetchKey(run) {
        const manifest = run?.deliverable_manifest && typeof run.deliverable_manifest === "object"
            ? run.deliverable_manifest
            : { primary_artifact_id: "", items: [] };
        const items = Array.isArray(manifest.items) ? manifest.items : [];
        return JSON.stringify({
            status: String(run?.status || "").trim(),
            finished_at: run?.finished_at || "",
            current_step_index: Number(run?.current_step_index || 0),
            primary_artifact_id: manifest.primary_artifact_id || "",
            items: items.map((item) => ({
                artifact_id: item?.artifact_id || "",
                role: item?.role || "",
                format: item?.format || "",
                mime_type: item?.mime_type || "",
            })),
        });
    }

    async function refreshConversationArtifactsForRootRuns(rootRuns, targetSessionId = sessionId) {
        const activeRunIds = new Set(rootRuns.map((run) => run.id));
        Array.from(conversationArtifactsByRunId.keys()).forEach((runId) => {
            if (!activeRunIds.has(runId)) {
                conversationArtifactsByRunId.delete(runId);
                conversationArtifactFetchKeys.delete(runId);
            }
        });

        const runsToFetch = rootRuns.filter((run) => {
            const nextKey = buildConversationArtifactFetchKey(run);
            if (RUN_ACTIVE_STATUSES.has(String(run?.status || "").trim())) {
                return true;
            }
            return !conversationArtifactsByRunId.has(run.id) || conversationArtifactFetchKeys.get(run.id) !== nextKey;
        });

        if (!runsToFetch.length) {
            return;
        }

        const fetchedEntries = await Promise.all(
            runsToFetch.map(async (run) => ({
                runId: run.id,
                fetchKey: buildConversationArtifactFetchKey(run),
                artifacts: await fetchJson(`/api/runs/${encodeURIComponent(run.id)}/artifacts`).catch(() => null),
            }))
        );

        if (targetSessionId !== sessionId) {
            return;
        }

        fetchedEntries.forEach((entry) => {
            if (Array.isArray(entry.artifacts)) {
                replaceConversationArtifactsForRun(entry.runId, entry.artifacts, {
                    updateFetchKey: true,
                });
            } else if (!conversationArtifactsByRunId.has(entry.runId)) {
                replaceConversationArtifactsForRun(entry.runId, [], {
                    updateFetchKey: true,
                });
            }
            conversationArtifactFetchKeys.set(entry.runId, entry.fetchKey);
        });
    }

    function syncLatestAssistantArtifactShelfFromSelectedRun(targetRunId = selectedRunId) {
        const rootRunId = resolveRootRunId(targetRunId);
        const rootRun = rootRunId ? getRunById(rootRunId) : null;
        const cachedArtifacts = rootRunId ? getCachedConversationArtifacts(rootRunId) : [];
        const artifacts = cachedArtifacts.length
            ? cachedArtifacts
            : selectedRunArtifacts.length
            ? selectedRunArtifacts
            : [];
        if (rootRun && artifacts.length) {
            const container =
                findAssistantStreamForRunId(rootRun.id) ||
                (targetRunId === activeRunId ? findLatestAssistantStream() : null);
            if (container) {
                setAssistantStreamRunId(container, rootRun.id);
                syncAssistantArtifactShelfWithRun(container, rootRun, artifacts);
            }
        }
        syncConversationArtifactShelvesWithRuns();
    }

    function buildRunCompletionSnapshot(run) {
        if (!run || run.status !== "completed") {
            return "";
        }
        const primaryArtifact = findPrimaryDeliverableArtifact(run);
        const latestOutputSummary = [...selectedRunSteps]
            .reverse()
            .find((step) => String(step?.output_summary || "").trim());
        const summaryText =
            String(primaryArtifact?.summary || "").trim() ||
            String(latestOutputSummary?.output_summary || "").trim() ||
            "本次运行已完成，主交付物已经生成。";
        const primaryDescriptor = createArtifactFileDescriptor(primaryArtifact);
        const actionButtons = primaryDescriptor
            ? renderFileActionButtons(primaryDescriptor, {
                buttonClassName:
                    "border-emerald-400/20 bg-black/20 text-on-surface hover:border-emerald-300/40 hover:text-emerald-200",
            })
            : "";
        const primaryName = primaryArtifact
            ? escapeHtml(primaryArtifact.display_name || primaryArtifact.uri || primaryArtifact.id)
            : "本次运行";

        return `
            <div class="rounded-2xl border border-emerald-400/20 bg-emerald-400/10 px-4 py-4">
                <div class="flex items-start justify-between gap-3">
                    <div class="min-w-0">
                        <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-emerald-300">完成摘要</div>
                        <div class="mt-2 text-sm leading-relaxed text-on-surface">${escapeHtml(truncate(summaryText, 220))}</div>
                        <div class="mt-2 text-[11px] text-emerald-100/90 break-all">主交付物：${primaryName}</div>
                    </div>
                    <span class="inline-flex items-center rounded-full border border-emerald-300/20 bg-black/20 px-2 py-1 text-[10px] font-mono uppercase tracking-[0.18em] text-emerald-200">
                        已交付
                    </span>
                </div>
                ${actionButtons ? `<div class="mt-4">${actionButtons}</div>` : ""}
            </div>
        `;
    }

    function renderDeliverableSection(run) {
        const manifest = run?.deliverable_manifest && typeof run.deliverable_manifest === "object"
            ? run.deliverable_manifest
            : { primary_artifact_id: null, items: [] };
        const items = Array.isArray(manifest.items) ? manifest.items : [];
        if (!items.length) {
            return `<div class="rounded-xl border border-dashed border-outline-variant/20 bg-surface-container-high px-3 py-3 text-sm text-on-surface-variant">当前还没有被标记为正式交付物的文件。</div>`;
        }

        const artifactMap = new Map(
            selectedRunArtifacts.map((artifact) => [artifact.id, artifact])
        );

        return items
            .slice(0, 4)
            .map((item) => {
                const artifact = artifactMap.get(item.artifact_id) || {
                    ...item,
                    id: item.artifact_id,
                    artifact_type: "document",
                    preview_kind: item.format || "none",
                };
                const isPrimary = item.artifact_id === manifest.primary_artifact_id;
                return renderArtifactCard(artifact, {
                    headingLabel: isPrimary ? "主交付物" : "补充交付物",
                    emphasize: isPrimary,
                });
            })
            .join("");
    }

    function renderRunDetail() {
        if (!$runDetailPanel || !$runDetailPill) return;
        if (!isFeatureEnabled("enable_durable_runs")) {
            $runDetailPill.classList.add("hidden");
            $runDetailPanel.innerHTML = `
                <div class="rounded-2xl border border-dashed border-outline-variant/20 bg-surface-container-low px-4 py-5 text-sm text-on-surface-variant">
                    当前环境未开启执行过程查看功能。
                </div>
            `;
            return;
        }
        const run = getSelectedRun();
        if (!run) {
            $runDetailPill.classList.add("hidden");
            $runDetailPanel.innerHTML = `
                <div class="rounded-2xl border border-dashed border-outline-variant/20 bg-surface-container-low px-4 py-5 text-sm text-on-surface-variant">
                    先从“最近记录”中选择一项，才能查看详细过程。
                </div>
            `;
            return;
        }

        const meta = getRunStatusMeta(run.status);
        const childRuns = getRunChildren(run.id);
        const recentSteps = selectedRunSteps.slice(-5);
        const recentTimeline = selectedRunTimeline.slice(-6);
        const completionSnapshot = buildRunCompletionSnapshot(run);
        const deliverablesContent = renderDeliverableSection(run);
        const recentStepsContent = recentSteps.length
            ? recentSteps
                  .map((step) => {
                      const stepMeta = getRunStatusMeta(
                          step.status === "failed"
                              ? "failed"
                              : step.status === "completed"
                                  ? "completed"
                                  : "running"
                      );
                      return `
                          <div class="mb-2 rounded-xl border border-outline-variant/10 bg-surface-container-high px-3 py-3">
                              <div class="flex items-start justify-between gap-3">
                                  <div class="min-w-0">
                                      <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">${escapeHtml(step.step_type)}</div>
                                      <div class="mt-1 text-sm text-on-surface">${escapeHtml(step.title || step.input_summary || "未命名步骤")}</div>
                                  </div>
                                  <span class="inline-flex items-center rounded-full border px-2 py-1 text-[10px] font-mono uppercase tracking-[0.18em] ${stepMeta.tone}">${escapeHtml(stepMeta.label)}</span>
                              </div>
                              ${
                                  step.output_summary
                                      ? `<div class="mt-2 text-[11px] text-on-surface-variant">${escapeHtml(truncate(step.output_summary, 220))}</div>`
                                      : ""
                              }
                              ${
                                  step.error_summary
                                      ? `<div class="mt-2 text-[11px] text-error">${escapeHtml(truncate(step.error_summary, 220))}</div>`
                                      : ""
                              }
                          </div>
                      `;
                  })
                  .join("")
            : `<div class="rounded-xl border border-dashed border-outline-variant/20 bg-surface-container-high px-3 py-3 text-sm text-on-surface-variant">当前还没有可查看的步骤记录。</div>`;
        const traceContent = !isFeatureEnabled("enable_run_trace")
            ? `<div class="rounded-xl border border-dashed border-outline-variant/20 bg-surface-container-high px-3 py-3 text-sm text-on-surface-variant">当前环境未开启过程时间线功能。</div>`
            : recentTimeline.length
                ? recentTimeline
                      .map(
                          (item) => `
                              <div class="mb-2 rounded-xl border border-outline-variant/10 bg-surface-container-high px-3 py-3">
                                  <div class="flex items-center justify-between gap-3">
                                      <div class="min-w-0">
                                          <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-primary">${escapeHtml(item.event_type || "过程事件")}</div>
                                          <div class="mt-1 text-sm text-on-surface">${escapeHtml(item.title || item.summary || item.payload_summary || "过程事件")}</div>
                                      </div>
                                      <div class="text-[10px] font-mono text-on-surface-variant">${escapeHtml(formatTimestamp(item.created_at || item.timestamp || ""))}</div>
                                  </div>
                                  ${
                                      item.summary && item.title !== item.summary
                                          ? `<div class="mt-2 text-[11px] text-on-surface-variant">${escapeHtml(truncate(item.summary, 220))}</div>`
                                          : ""
                                  }
                              </div>
                          `
                      )
                      .join("")
                : `<div class="rounded-xl border border-dashed border-outline-variant/20 bg-surface-container-high px-3 py-3 text-sm text-on-surface-variant">当前还没有可查看的过程时间线。</div>`;
        const childTreeContent = renderRunDagSection();
        const artifactsContent = selectedRunArtifacts.length
            ? selectedRunArtifacts
                  .slice(0, 4)
                  .map((artifact) => renderArtifactCard(artifact, {
                      headingLabel: formatArtifactTypeLabel(artifact.artifact_type),
                  }))
                  .join("")
            : `<div class="rounded-xl border border-dashed border-outline-variant/20 bg-surface-container-high px-3 py-3 text-sm text-on-surface-variant">当前没有可查看的产出。</div>`;
        const approvalsContent = !isFeatureEnabled("enable_approval_flow")
            ? `<div class="rounded-xl border border-dashed border-outline-variant/20 bg-surface-container-high px-3 py-3 text-sm text-on-surface-variant">当前环境未开启审批流程功能。</div>`
            : selectedRunApprovals.length
                ? selectedRunApprovals.map((approval) => renderApprovalRecord(approval)).join("")
                : `<div class="rounded-xl border border-dashed border-outline-variant/20 bg-surface-container-high px-3 py-3 text-sm text-on-surface-variant">当前没有审批记录。</div>`;

        $runDetailPill.classList.remove("hidden");
        $runDetailPill.textContent = run.id.slice(0, 8);

        $runDetailPanel.innerHTML = `
            <article class="rounded-2xl border border-outline-variant/10 bg-surface-container-high overflow-hidden">
                <div class="border-b border-outline-variant/10 px-4 py-4">
                    <div class="flex items-start justify-between gap-3">
                        <div class="min-w-0">
                            <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">已选任务</div>
                            <h3 class="mt-2 text-sm font-bold text-on-surface leading-relaxed">${escapeHtml(run.goal || "未命名任务")}</h3>
                            <p class="mt-2 text-[11px] text-on-surface-variant">${escapeHtml(buildRunMetaLine(run))}</p>
                        </div>
                        <span class="inline-flex items-center gap-1 rounded-full border px-2 py-1 text-[10px] font-mono uppercase tracking-[0.18em] ${meta.tone}">
                            <span class="material-symbols-outlined text-sm">${meta.icon}</span>
                            <span>${escapeHtml(meta.label)}</span>
                        </span>
                    </div>
                    <div class="mt-4 grid grid-cols-2 gap-3 text-[10px] font-mono uppercase tracking-[0.18em]">
                        <div class="rounded-xl border border-outline-variant/10 bg-black/20 px-3 py-2">
                            <div class="text-on-surface-variant">执行助手</div>
                            <div class="mt-1 text-on-surface">${escapeHtml(getRunAgentName(run))}</div>
                        </div>
                        <div class="rounded-xl border border-outline-variant/10 bg-black/20 px-3 py-2">
                            <div class="text-on-surface-variant">子任务</div>
                            <div class="mt-1 text-on-surface">${escapeHtml(String(childRuns.length))}</div>
                        </div>
                        <div class="rounded-xl border border-outline-variant/10 bg-black/20 px-3 py-2">
                            <div class="text-on-surface-variant">产出</div>
                            <div class="mt-1 text-on-surface">${escapeHtml(String(selectedRunArtifacts.length))}</div>
                        </div>
                        <div class="rounded-xl border border-outline-variant/10 bg-black/20 px-3 py-2">
                            <div class="text-on-surface-variant">审批</div>
                            <div class="mt-1 text-on-surface">${
                                isFeatureEnabled("enable_approval_flow")
                                    ? escapeHtml(String(selectedRunApprovals.length))
                                    : "未开启"
                            }</div>
                        </div>
                    </div>
                    ${
                        run.error_summary
                            ? `<div class="mt-4 rounded-xl border border-error/20 bg-error/10 px-3 py-3 text-[11px] text-error/90">
                                <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-error">失败说明</div>
                                <div class="mt-2 leading-relaxed">${escapeHtml(run.error_summary)}</div>
                              </div>`
                            : ""
                    }
                    <div class="mt-4 flex flex-wrap gap-2">
                        ${
                            isRunRecoverable(run)
                                ? `<button data-run-action="resume" data-run-id="${escapeHtml(run.id)}" class="inline-flex items-center gap-2 rounded-xl border border-primary/30 bg-primary/10 px-3 py-2 text-[10px] font-mono uppercase tracking-[0.18em] text-primary hover:bg-primary/15 transition-colors">
                                    <span class="material-symbols-outlined text-sm">restart_alt</span>
                                    <span>继续执行</span>
                                   </button>`
                                : ""
                        }
                        ${
                            isRunActive(run)
                                ? `<button data-run-action="cancel" data-run-id="${escapeHtml(run.id)}" class="inline-flex items-center gap-2 rounded-xl border border-error/20 bg-error/10 px-3 py-2 text-[10px] font-mono uppercase tracking-[0.18em] text-error hover:bg-error/15 transition-colors">
                                    <span class="material-symbols-outlined text-sm">stop_circle</span>
                                    <span>取消执行</span>
                                   </button>`
                                : ""
                        }
                    </div>
                </div>

                <div class="px-4 py-4 space-y-3">
                    ${completionSnapshot}
                    ${renderInspectorSection("交付结果", deliverablesContent, { open: true })}
                    ${renderInspectorSection("步骤记录", recentStepsContent, { open: true })}
                    ${renderInspectorSection("过程时间线", traceContent)}
                    ${renderInspectorSection("子任务树", childTreeContent)}
                    ${renderInspectorSection("产出", artifactsContent)}
                    ${renderInspectorSection("审批", approvalsContent)}
                </div>
            </article>
        `;
    }

    function renderRunPanels() {
        renderRunSessionSummary();
        renderApprovalInbox();
        renderRunActiveCard();
        renderRunHistory();
        renderRunDetail();
        renderLearningPanel();
        renderRunBanner();
        updateComposerActionButton();
    }

    function clearRunState() {
        sessionRuns = [];
        selectedRunId = null;
        activeRunId = null;
        selectedRunSteps = [];
        selectedRunTimeline = [];
        selectedRunArtifacts = [];
        selectedRunApprovals = [];
        sessionPendingApprovals = [];
        selectedRunTree = null;
        resetConversationArtifactCache();
        renderRunPanels();
    }

    function clearRunRefreshTimer() {
        if (!runRefreshTimer) return;
        clearTimeout(runRefreshTimer);
        runRefreshTimer = null;
    }

    function scheduleRunRefresh() {
        clearRunRefreshTimer();
        if (!isFeatureEnabled("enable_durable_runs")) {
            return;
        }
        const shouldPoll =
            Boolean(sessionId) &&
            (isStreaming || getRootRuns().some((run) => isRunActive(run)));
        if (!shouldPoll) {
            return;
        }
        runRefreshTimer = setTimeout(() => {
            refreshRunSidebar().catch((error) => {
                console.error("Failed to refresh run panel:", error);
            });
        }, 1500);
    }

    async function refreshRunSidebar({ focusRunId = null, preserveSelection = true } = {}) {
        if (!isFeatureEnabled("enable_durable_runs") || !sessionId) {
            clearRunState();
            clearRunRefreshTimer();
            return;
        }

        const targetSessionId = sessionId;
        const params = new URLSearchParams({
            session_id: targetSessionId,
            limit: "100",
        });
        const runs = await fetchJson(`/api/runs?${params.toString()}`);
        if (targetSessionId !== sessionId) {
            return;
        }

        sessionRuns = Array.isArray(runs) ? runs : [];
        const sessionRunIds = new Set(sessionRuns.map((run) => run.id));
        const rootRunList = getRootRuns();
        const currentRun = getCurrentRootRun();
        activeRunId = currentRun?.id || null;

        const pendingApprovals = isFeatureEnabled("enable_approval_flow")
            ? await fetchJson(
                  `/api/approvals?status=pending&session_id=${encodeURIComponent(targetSessionId)}`
              ).catch(() => [])
            : [];
        if (targetSessionId !== sessionId) {
            return;
        }
        sessionPendingApprovals = Array.isArray(pendingApprovals)
            ? pendingApprovals.filter((approval) => sessionRunIds.has(approval.run_id))
            : [];

        const rootRuns = getRootRunsChronologically();
        await refreshConversationArtifactsForRootRuns(rootRuns, targetSessionId);
        if (targetSessionId !== sessionId) {
            return;
        }

        const nextSelectedRunId =
            focusRunId && sessionRuns.some((run) => run.id === focusRunId)
                ? focusRunId
                : preserveSelection && selectedRunId && sessionRuns.some((run) => run.id === selectedRunId)
                    ? selectedRunId
                    : activeRunId || rootRunList[0]?.id || null;

        selectedRunId = nextSelectedRunId;

        if (selectedRunId) {
            const [steps, timeline, artifacts, approvals, tree] = await Promise.all([
                fetchJson(`/api/runs/${selectedRunId}/steps`),
                isFeatureEnabled("enable_run_trace")
                    ? fetchJson(`/api/runs/${selectedRunId}/trace/timeline`).catch(() => [])
                    : Promise.resolve([]),
                fetchJson(`/api/runs/${selectedRunId}/artifacts`),
                isFeatureEnabled("enable_approval_flow")
                    ? fetchJson(`/api/approvals?run_id=${encodeURIComponent(selectedRunId)}`)
                    : Promise.resolve([]),
                isFeatureEnabled("enable_run_trace")
                    ? fetchJson(`/api/runs/${selectedRunId}/trace/tree`).catch(() => null)
                    : Promise.resolve(null),
            ]);
            if (targetSessionId !== sessionId) {
                return;
            }
            selectedRunSteps = Array.isArray(steps) ? steps : [];
            selectedRunTimeline = Array.isArray(timeline) ? timeline : [];
            selectedRunArtifacts = Array.isArray(artifacts) ? artifacts : [];
            selectedRunApprovals = Array.isArray(approvals) ? approvals : [];
            selectedRunTree = tree && typeof tree === "object" ? tree : null;
            const selectedRun = getRunById(selectedRunId);
            if (selectedRun) {
                if (selectedRun.parent_run_id) {
                    mergeConversationArtifactsForRun(selectedRun.id, selectedRunArtifacts);
                } else {
                    replaceConversationArtifactsForRun(selectedRun.id, selectedRunArtifacts, {
                        updateFetchKey: true,
                    });
                }
            }
        } else {
            selectedRunSteps = [];
            selectedRunTimeline = [];
            selectedRunArtifacts = [];
            selectedRunApprovals = [];
            selectedRunTree = null;
        }

        renderRunPanels();
        syncConversationArtifactShelvesWithRuns();
        scheduleRunRefresh();
    }

    async function handleRunAction(action, runId = selectedRunId) {
        if (!isFeatureEnabled("enable_durable_runs")) {
            return;
        }
        if (!runId && action !== "refresh") return;

        if (action === "select") {
            setSidebarTab("runs");
            selectedRunId = runId;
            await refreshRunSidebar({ focusRunId: runId, preserveSelection: false });
            return;
        }

        if (action === "refresh") {
            await refreshRunSidebar({ focusRunId: runId || activeRunId });
            return;
        }

        if (action === "cancel") {
            await fetchJson(`/api/runs/${runId}/cancel`, { method: "POST" });
            await refreshRunSidebar({ focusRunId: runId });
            return;
        }

        if (action === "resume") {
            await fetchJson(`/api/runs/${runId}/resume`, { method: "POST" });
            await refreshRunSidebar({ focusRunId: runId });
        }
    }

    async function handleApprovalAction(action, approvalId, runId = selectedRunId) {
        if (!isFeatureEnabled("enable_approval_flow")) {
            return;
        }
        const actionMap = {
            "grant-once": {
                endpoint: "grant",
                decisionScope: "once",
                notes: "Granted for this tool call from workspace approval inbox.",
            },
            "grant-run": {
                endpoint: "grant",
                decisionScope: "run",
                notes: "Granted for the current run from workspace approval inbox.",
            },
            "grant-template": {
                endpoint: "grant",
                decisionScope: "template",
                notes: "Granted and persisted into the agent approval policy from workspace approval inbox.",
            },
            deny: {
                endpoint: "deny",
                decisionScope: "once",
                notes: "Denied from workspace approval inbox.",
            },
        };
        const config = actionMap[action];
        if (!approvalId || !config) {
            return;
        }
        if (approvalActionIds.has(approvalId)) {
            return;
        }

        approvalActionIds.add(approvalId);
        renderRunPanels();

        try {
            await fetchJson(`/api/approvals/${approvalId}/${config.endpoint}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    decision_notes: config.notes,
                    decision_scope: config.decisionScope,
                }),
            });
            await refreshRunSidebar({ focusRunId: runId || selectedRunId });
        } catch (error) {
            setStatus(error.message || "Approval action failed");
            throw error;
        } finally {
            approvalActionIds.delete(approvalId);
            renderRunPanels();
        }
    }

    function renderSharedContextPanel() {
        if (!$sharedContextSidebar || !$sharedContextList || !$sharedContextCount) {
            return;
        }

        $sharedContextCount.textContent = String(sharedContextEntries.length);

        if (!sharedContextEntries.length) {
            $sharedContextList.innerHTML = '<div class="text-[10px] text-on-surface-variant italic">当前还没有可查看的会话知识。</div>';
            return;
        }

        $sharedContextList.innerHTML = "";

        const fragment = document.createDocumentFragment();
        sharedContextEntries.forEach((entry) => {
            const item = document.createElement("article");
            item.className = "p-3 bg-surface-container-high rounded-xl border border-outline-variant/10";
            item.innerHTML = `
                <div class="flex items-center justify-between mb-1">
                    <span class="text-[11px] font-bold text-on-surface uppercase tracking-wider">${escapeHtml(entry.title || entry.id || "知识节点")}</span>
                    <span class="text-[9px] text-emerald-500 font-mono tracking-widest uppercase truncate max-w-[80px]" title="${escapeHtml(entry.category || "通用")}">${escapeHtml(entry.category || "通用")}</span>
                </div>
                <div class="text-[10px] text-on-surface-variant/80 font-mono mb-1 truncate">${escapeHtml(entry.source || "未知来源")} / ${escapeHtml(formatTimestamp(entry.timestamp))}</div>
                <div class="text-[11px] text-on-surface-variant leading-relaxed overflow-hidden" style="display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical;">${escapeAndFormat(entry.content || "")}</div>
            `;
            fragment.appendChild(item);
        });

        $sharedContextList.appendChild(fragment);
    }

    function clearSharedContextPanel() {
        sharedContextEntries = [];
        renderSharedContextPanel();
    }

    // setSharedContextCollapsed removed in favor of generic toggle engine
    async function refreshSharedContextPanel(targetSessionId = sessionId) {
        if (!targetSessionId) {
            clearSharedContextPanel();
            return;
        }

        try {
            const data = await fetchJson(`/api/sessions/${targetSessionId}/shared-context?limit=200`);
            if (targetSessionId !== sessionId) {
                return;
            }
            sharedContextEntries = Array.isArray(data.entries) ? data.entries : [];
            renderSharedContextPanel();
        } catch (error) {
            console.error("Failed to load shared context:", error);
            if (targetSessionId === sessionId) {
                clearSharedContextPanel();
            }
        }
    }

    function scheduleSharedContextRefresh() {
        if (!sessionId) return;
        if (sharedContextRefreshTimer) {
            clearTimeout(sharedContextRefreshTimer);
        }
        sharedContextRefreshTimer = setTimeout(() => {
            refreshSharedContextPanel(sessionId);
        }, 120);
    }

    function hasShareContextUpdate(event) {
        if (!event || typeof event !== "object") {
            return false;
        }

        if (event.type === "tool_result") {
            const data = event.data || {};
            return data.name === "share_context" && data.success === true;
        }

        if (event.type === "sub_task") {
            return hasShareContextUpdate(event.data?.event);
        }

        return false;
    }

    function sessionGroupStorageKey(agentId) {
        return `session_group_collapsed_${normalizeAgentId(agentId)}`;
    }

    function isSessionGroupCollapsed(agentId) {
        return localStorage.getItem(sessionGroupStorageKey(agentId)) === "true";
    }

    function setSessionGroupCollapsed(agentId, collapsed) {
        localStorage.setItem(sessionGroupStorageKey(agentId), collapsed ? "true" : "false");
    }

    function groupSessionsByAgent(sessionItems) {
        const groups = new Map();
        sessionItems.forEach((session) => {
            const agentId = normalizeAgentId(session.agent_id);
            if (!groups.has(agentId)) {
                groups.set(agentId, []);
            }
            groups.get(agentId).push(session);
        });
        return groups;
    }

    function createSessionGroup(agentId, agentSessions) {
        const group = document.createElement("section");
        const agentName = getAgentDisplayName(agentId);
        const collapsed = isSessionGroupCollapsed(agentId);
        const sessionCountLabel = `${agentSessions.length}`;
        const sessionWordLabel = agentSessions.length === 1 ? "session" : "sessions";

        group.className = "rounded-2xl border border-outline-variant/10 bg-surface-container-low/60 overflow-hidden";
        group.dataset.agentGroup = agentId;
        group.innerHTML = `
            <button
                type="button"
                data-toggle-agent-group="${escapeHtml(agentId)}"
                class="w-full px-4 py-3 flex items-center justify-between gap-3 bg-surface-container-highest/40 hover:bg-surface-container-highest/60 transition-colors border-b border-outline-variant/10"
            >
                <div class="min-w-0 flex items-center gap-3">
                    <span class="material-symbols-outlined text-sm text-secondary shrink-0">${collapsed ? "chevron_right" : "expand_more"}</span>
                    <div class="min-w-0 text-left">
                        <div class="text-[11px] font-bold uppercase tracking-[0.18em] text-secondary truncate">${escapeHtml(agentName)}</div>
                        <div class="text-[10px] font-mono text-on-surface-variant truncate">${escapeHtml(agentId)}</div>
                    </div>
                </div>
                <div class="shrink-0 px-2 py-1 rounded-full bg-surface border border-outline-variant/10 text-[10px] font-mono uppercase tracking-widest text-on-surface-variant">
                    ${sessionCountLabel} ${sessionWordLabel}
                </div>
            </button>
        `;

        const body = document.createElement("div");
        body.className = `${collapsed ? "hidden " : ""}p-2 space-y-2`;
        body.dataset.agentGroupBody = agentId;

        agentSessions.forEach((session) => {
            body.appendChild(createSessionItem(session));
        });

        group.appendChild(body);
        return group;
    }

    function createSessionItem(session) {
        const item = document.createElement("div");
        const isActive = session.session_id === sessionId;
        item.className = `p-3 rounded-xl group cursor-pointer transition-colors ${isActive ? 'bg-surface-bright border-l-2 border-secondary shadow-[inset_0_0_0_1px_rgba(213,117,255,0.18)]' : 'hover:bg-surface-container-high'}`;
        item.dataset.sessionId = session.session_id;
        item.innerHTML = `
            <div class="flex justify-between items-start mb-1">
                <span class="text-xs font-mono ${isActive ? 'text-primary' : 'text-on-surface-variant'} truncate mr-2" title="${escapeHtml(session.session_id)}">${escapeHtml(session.session_id.substring(0, 8))}</span>
                <button class="text-xs flex items-center justify-center w-5 h-5 rounded hover:bg-error/20 hover:text-error opacity-0 group-hover:opacity-100 transition-opacity" type="button" data-delete-session="${session.session_id}" title="删除会话">×</button>
            </div>
            <p class="text-sm font-medium line-clamp-1 ${isActive ? '' : 'text-on-surface-variant group-hover:text-on-surface'}">${escapeHtml(session.title || "新对话")}</p>
            <div class="text-[10px] text-on-surface-variant font-mono mt-1">${escapeHtml(formatTimestamp(session.updated_at))}</div>
        `;
        return item;
    }

    function renderSessionList() {
        $sessionList.innerHTML = "";
        if (!sessions.length) {
            const empty = document.createElement("div");
            empty.className = "empty-sessions";
            empty.textContent = "还没有历史会话。创建一个新对话后，这里会自动保存。";
            $sessionList.appendChild(empty);
            return;
        }

        const fragment = document.createDocumentFragment();
        const groupedSessions = groupSessionsByAgent(sessions);

        groupedSessions.forEach((agentSessions, agentId) => {
            fragment.appendChild(createSessionGroup(agentId, agentSessions));
        });

        $sessionList.appendChild(fragment);
    }

    async function fetchJson(url, options = {}) {
        const { allow404 = false, ...fetchOptions } = options;
        const response = await fetch(url, fetchOptions);
        if (response.status === 401) {
            // 未登录，跳转到登录页
            const currentPath = encodeURIComponent(window.location.href);
            window.location.href = `/static/login.html?redirect=${currentPath}`;
            // 抛错阻止后续代码继续执行
            throw new Error("Authentication required. Redirecting to login page.");
        }
        if (allow404 && response.status === 404) {
            return null;
        }
        if (!response.ok) {
            let message = `HTTP ${response.status}`;
            try {
                const data = await response.json();
                if (typeof data?.detail === "string" && data.detail) {
                    message = data.detail;
                }
            } catch {
                // Keep the fallback HTTP status text.
            }
            throw new Error(message);
        }
        return response.json();
    }

    const claviAgentWorkspaceApi = {
        fetchJson,
        setStatus,
        getAgentDisplayName,
    };
    window.ClaviAgentWorkspaceApi = claviAgentWorkspaceApi;
    window.MiniAgentWorkspaceApi = claviAgentWorkspaceApi;

    async function refreshSessionList(activeId = sessionId) {
        sessions = await fetchJson("/api/sessions");
        if (activeId && sessions.some((item) => item.session_id === activeId)) {
            sessionId = activeId;
        }
        renderSessionList();
    }

    async function createSession({ openChat = true, agent_id = null } = {}) {
        setStatus("正在创建新会话");
        setComposerState("Loading");

        const payload = agent_id ? { agent_id } : {};

        const data = await fetchJson("/api/sessions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        sessionId = data.session_id;
        setActiveAgent(data.agent_id);
        setSessionGroupCollapsed(data.agent_id, false);
        await refreshSessionList(sessionId);
        $messages.innerHTML = "";
        clearComposerAttachments();
        clearRunState();
        clearSharedContextPanel();
        clearSessionUploadsPanel();
        if (openChat) {
            switchView("orchestrator"); // Using the generic router rather than relying on legacy showChatView
        }
        setStatus("会话已就绪");
        setComposerState("Ready");
        if (isFeatureEnabled("enable_durable_runs")) {
            await refreshRunSidebar({ preserveSelection: false });
        }
        await refreshSessionUploads(sessionId);
        return data;
    }

    // --- Agent Marketplace Functions ---

    function normalizeSkillInstallStatus(status) {
        return status && typeof status === "object"
            ? status
            : { state: "idle", message: "No skill installation in progress.", packages: [], error: "" };
    }

    function isSkillInstallActive(status) {
        const state = normalizeSkillInstallStatus(status).state;
        return state === "queued" || state === "running";
    }

    function formatSkillInstallStatus(status) {
        const normalized = normalizeSkillInstallStatus(status);
        const packageCount = Array.isArray(normalized.packages) ? normalized.packages.length : 0;
        const suffix = packageCount > 0 ? ` (${packageCount} package${packageCount > 1 ? "s" : ""})` : "";

        if (normalized.state === "queued") return `Queued${suffix}`;
        if (normalized.state === "running") return `Installing${suffix}`;
        if (normalized.state === "succeeded") return `Installed${suffix}`;
        if (normalized.state === "failed") return `Failed${suffix}`;
        return "Idle";
    }

    function getSkillInstallPresentation(status) {
        const normalized = normalizeSkillInstallStatus(status);
        const label = formatSkillInstallStatus(normalized);
        const toneClass =
            normalized.state === "failed"
                ? "text-error border-error/30 bg-error/10"
                : isSkillInstallActive(normalized)
                    ? "text-primary border-primary/30 bg-primary/10"
                    : normalized.state === "succeeded"
                        ? "text-emerald-400 border-emerald-400/30 bg-emerald-400/10"
                        : "text-on-surface-variant border-outline-variant/20 bg-surface-container-highest";
        const dotClass =
            normalized.state === "failed"
                ? "bg-error"
                : isSkillInstallActive(normalized)
                    ? "bg-primary"
                    : normalized.state === "succeeded"
                        ? "bg-emerald-400"
                        : "bg-outline-variant";
        return { normalized, label, toneClass, dotClass };
    }

    function renderSkillStatusSummary(status, skillCount, options = {}) {
        const { showMessage = false, compact = false } = options;
        const { normalized, label, toneClass, dotClass } = getSkillInstallPresentation(status);
        const summaryText = `Skills: ${skillCount}`;
        const message = normalized.error || normalized.message || "No skill installation in progress.";
        const messageClass = normalized.state === "failed" ? "text-error/90" : "text-on-surface-variant";
        const wrapperClass = compact ? "gap-2" : "gap-3";

        return `
            <div class="flex flex-wrap items-center ${wrapperClass}">
                <span class="text-[11px] font-mono text-on-surface-variant">${escapeHtml(summaryText)}</span>
                <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-[10px] font-bold uppercase tracking-[0.18em] ${toneClass}">
                    <span class="w-1.5 h-1.5 rounded-full ${dotClass}"></span>
                    ${escapeHtml(label)}
                </span>
            </div>
            ${showMessage ? `<div class="mt-2 text-xs ${messageClass}">${escapeHtml(message)}</div>` : ""}
        `;
    }

    function renderSkillInstallStatus(status, skillCount = 0) {
        const container = document.getElementById("skillInstallStatus");
        if (!container) return;

        container.className = "mt-3";
        container.innerHTML = renderSkillStatusSummary(status, skillCount, { showMessage: true });
    }

    function clearAgentGridPoll() {
        if (!agentGridPollTimer) return;
        clearTimeout(agentGridPollTimer);
        agentGridPollTimer = null;
    }

    function scheduleAgentGridPoll() {
        clearAgentGridPoll();
        agentGridPollTimer = setTimeout(() => {
            window.renderAgentsGrid({ background: true });
        }, 3000);
    }

    function splitCommaSeparatedValues(rawValue) {
        return String(rawValue || "")
            .split(/[\n,]/)
            .map((item) => item.trim())
            .filter(Boolean);
    }

    function joinCommaSeparatedValues(values) {
        return Array.isArray(values) ? values.filter(Boolean).join(", ") : "";
    }

    function normalizeOptionalBooleanSelectValue(value) {
        return typeof value === "boolean" ? String(value) : "";
    }

    function parseOptionalBooleanSelectValue(value) {
        if (value === "true") return true;
        if (value === "false") return false;
        return null;
    }

    function getDelegationModeMeta(mode) {
        if (mode === "supervisor_only") {
            return {
                label: "Supervisor Only",
                summary: "主 agent 只做理解、规划、调度与验收，具副作用的执行默认交给 worker。",
                tone: "border-amber-400/20 bg-amber-400/10 text-amber-200",
            };
        }
        if (mode === "hybrid") {
            return {
                label: "Hybrid",
                summary: "主 agent 可以直做，也可以委派；适合逐步迁移阶段。",
                tone: "border-secondary/20 bg-secondary/10 text-secondary",
            };
        }
        return {
            label: "Prefer Delegate",
            summary: "主 agent 默认优先委派执行类工作，必要时保留少量直做路径。",
            tone: "border-primary/20 bg-primary/10 text-primary",
        };
    }

    function buildRoutingSummary() {
        return "主 agent / 子 agent 的 API 组与模型路由统一继承账号 API Settings。";
    }

    function normalizeTemplatePolicies(agent) {
        const workspacePolicy = agent?.workspace_policy || {};
        const approvalPolicy = agent?.approval_policy || {};
        const runPolicy = agent?.run_policy || {};
        const delegationPolicy = agent?.delegation_policy || {};
        const tools = Array.isArray(agent?.tools) && agent.tools.length
            ? agent.tools
            : defaultAgentToolset;

        const parsedTimeout = Number.parseInt(runPolicy.timeout_seconds, 10);
        const parsedConcurrency = Number.parseInt(runPolicy.max_concurrent_runs, 10);

        return {
            tools,
            workspacePolicy: {
                mode: workspacePolicy.mode || agent?.workspace_type || "isolated",
                allow_session_override: workspacePolicy.allow_session_override !== false,
                readable_roots: Array.isArray(workspacePolicy.readable_roots)
                    ? workspacePolicy.readable_roots
                    : [],
                writable_roots: Array.isArray(workspacePolicy.writable_roots)
                    ? workspacePolicy.writable_roots
                    : [],
                read_only_tools: Array.isArray(workspacePolicy.read_only_tools)
                    ? workspacePolicy.read_only_tools
                    : [],
                disabled_tools: Array.isArray(workspacePolicy.disabled_tools)
                    ? workspacePolicy.disabled_tools
                    : [],
                allowed_shell_command_prefixes: Array.isArray(workspacePolicy.allowed_shell_command_prefixes)
                    ? workspacePolicy.allowed_shell_command_prefixes
                    : [],
                allowed_network_domains: Array.isArray(workspacePolicy.allowed_network_domains)
                    ? workspacePolicy.allowed_network_domains
                    : [],
            },
            approvalPolicy: {
                mode: approvalPolicy.mode || "default",
                require_approval_tools: Array.isArray(approvalPolicy.require_approval_tools)
                    ? approvalPolicy.require_approval_tools
                    : [],
                auto_approve_tools: Array.isArray(approvalPolicy.auto_approve_tools)
                    ? approvalPolicy.auto_approve_tools
                    : [],
                require_approval_risk_levels: Array.isArray(approvalPolicy.require_approval_risk_levels)
                    ? approvalPolicy.require_approval_risk_levels
                    : [],
                require_approval_risk_categories: Array.isArray(approvalPolicy.require_approval_risk_categories)
                    ? approvalPolicy.require_approval_risk_categories
                    : [],
                notes: approvalPolicy.notes || "",
            },
            runPolicy: {
                timeout_seconds: Number.isFinite(parsedTimeout) && parsedTimeout > 0 ? parsedTimeout : null,
                max_concurrent_runs: Number.isFinite(parsedConcurrency) && parsedConcurrency > 0 ? parsedConcurrency : 1,
            },
            delegationPolicy: {
                mode: delegationPolicy.mode || "prefer_delegate",
                require_delegate_for_write_actions: Boolean(delegationPolicy.require_delegate_for_write_actions),
                require_delegate_for_shell: Boolean(delegationPolicy.require_delegate_for_shell),
                require_delegate_for_stateful_mcp: Boolean(delegationPolicy.require_delegate_for_stateful_mcp),
                allow_main_agent_read_tools: delegationPolicy.allow_main_agent_read_tools !== false,
                verify_worker_output: delegationPolicy.verify_worker_output !== false,
                prefer_batch_delegate: delegationPolicy.prefer_batch_delegate !== false,
            },
        };
    }

    function renderAgentDefaultToolset(tools) {
        const container = document.getElementById("agentDefaultToolsContainer");
        if (!container) return;

        const items = Array.isArray(tools) ? tools.filter(Boolean) : [];
        if (!items.length) {
            container.innerHTML = `<div class="text-sm text-on-surface-variant">Default workspace tools will be inherited when this template is saved.</div>`;
            return;
        }

        container.innerHTML = `
            <div class="flex flex-wrap gap-2">
                ${items.map((tool) => `
                    <span class="inline-flex items-center rounded-full border border-primary/20 bg-primary/10 px-2.5 py-1 text-[11px] font-mono text-primary">
                        ${escapeHtml(tool)}
                    </span>
                `).join("")}
            </div>
            <div class="mt-3 text-xs text-on-surface-variant">Inherited from the default workspace toolset so template configuration and runtime execution stay aligned.</div>
        `;
    }

    function renderAgentReadonlyBanner(agent) {
        const container = document.getElementById("agentReadonlyBanner");
        if (!container) return;
        if (!agent?.is_system) {
            container.classList.add("hidden");
            container.innerHTML = "";
            return;
        }

        const { delegationPolicy } = normalizeTemplatePolicies(agent);
        const modeMeta = getDelegationModeMeta(delegationPolicy.mode);
        const guardrails = [
            delegationPolicy.require_delegate_for_write_actions ? "写操作强制委派" : "写操作可按模式直做",
            delegationPolicy.require_delegate_for_shell ? "Shell 强制委派" : "Shell 按模式决策",
            delegationPolicy.require_delegate_for_stateful_mcp ? "有状态 MCP 强制委派" : "MCP 按模式决策",
            delegationPolicy.allow_main_agent_read_tools ? "主 agent 保留只读工具" : "主 agent 不保留只读工具",
            delegationPolicy.verify_worker_output ? "开启 worker 验收" : "关闭 worker 验收",
            delegationPolicy.prefer_batch_delegate ? "优先批量委派" : "可串行委派",
        ];

        container.classList.remove("hidden");
        container.innerHTML = `
            <div class="rounded-2xl border border-emerald-400/20 bg-emerald-400/10 p-4 space-y-3">
                <div class="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
                    <div>
                        <div class="text-[10px] font-bold uppercase tracking-[0.22em] text-emerald-300">系统模板只读展示</div>
                        <p class="mt-2 text-sm leading-relaxed text-on-surface">
                            该模板由系统维护，当前窗口仅用于查看默认调度策略。主 / 子 agent 的 API 组与模型路由请在 API Settings 中调整。
                        </p>
                    </div>
                    <span class="inline-flex items-center rounded-full border px-3 py-1 text-[10px] font-bold uppercase tracking-[0.18em] ${modeMeta.tone}">
                        ${escapeHtml(modeMeta.label)}
                    </span>
                </div>
                <div class="text-xs leading-relaxed text-on-surface-variant">${escapeHtml(modeMeta.summary)}</div>
                <div class="flex flex-wrap gap-2">
                    ${guardrails.map((item) => `
                        <span class="inline-flex items-center rounded-full border border-white/10 bg-surface-container px-2.5 py-1 text-[10px] font-mono text-on-surface-variant">
                            ${escapeHtml(item)}
                        </span>
                    `).join("")}
                </div>
                <div class="rounded-xl border border-outline-variant/10 bg-surface-container p-3 text-xs leading-relaxed text-on-surface-variant">
                    ${escapeHtml(buildRoutingSummary())}
                </div>
            </div>
        `;
    }

    function setAgentModalReadonlyState(isReadonly) {
        const editableIds = [
            "agentNameInput",
            "agentDescInput",
            "agentPromptInput",
            "agentWorkspaceModeInput",
            "agentApprovalModeInput",
            "agentAllowSessionOverrideInput",
            "agentRunTimeoutInput",
            "agentRunConcurrencyInput",
            "agentReadableRootsInput",
            "agentWritableRootsInput",
            "agentReadOnlyToolsInput",
            "agentDisabledToolsInput",
            "agentAllowedShellCommandsInput",
            "agentAllowedNetworkDomainsInput",
            "agentRequireApprovalToolsInput",
            "agentAutoApproveToolsInput",
            "agentRequireApprovalRiskLevelsInput",
            "agentRequireApprovalRiskCategoriesInput",
            "agentPolicyNotesInput",
            "agentDelegationModeInput",
            "agentRequireDelegateWriteInput",
            "agentRequireDelegateShellInput",
            "agentRequireDelegateMcpInput",
            "agentAllowMainReadToolsInput",
            "agentVerifyWorkerOutputInput",
            "agentPreferBatchDelegateInput",
            "skillSearchKeywordInput",
        ];
        editableIds.forEach((id) => {
            const element = document.getElementById(id);
            if (!element) return;
            element.disabled = isReadonly;
            element.classList.toggle("opacity-60", isReadonly);
            element.classList.toggle("cursor-not-allowed", isReadonly);
        });

        const brainstormBtn = document.getElementById("btnBrainstormAgent");
        const searchBtn = document.getElementById("btnSearchSkills");
        const installBtn = document.getElementById("btnInstallSelectedSkills");
        [brainstormBtn, searchBtn, installBtn].forEach((button) => {
            if (!button) return;
            if (isReadonly) {
                button.disabled = true;
                button.classList.add("opacity-50", "cursor-not-allowed");
            } else {
                button.classList.remove("opacity-50", "cursor-not-allowed");
            }
        });

        const saveBtn = document.getElementById("saveAgentButton");
        if (saveBtn) {
            saveBtn.classList.toggle("hidden", isReadonly);
        }
    }

    function populateAgentPolicyFields(agent) {
        const {
            tools,
            workspacePolicy,
            approvalPolicy,
            runPolicy,
            delegationPolicy,
        } = normalizeTemplatePolicies(agent);

        renderAgentDefaultToolset(tools);
        renderAgentReadonlyBanner(agent);

        const workspaceModeInput = document.getElementById("agentWorkspaceModeInput");
        const allowSessionOverrideInput = document.getElementById("agentAllowSessionOverrideInput");
        const readableRootsInput = document.getElementById("agentReadableRootsInput");
        const writableRootsInput = document.getElementById("agentWritableRootsInput");
        const readOnlyToolsInput = document.getElementById("agentReadOnlyToolsInput");
        const disabledToolsInput = document.getElementById("agentDisabledToolsInput");
        const allowedShellCommandsInput = document.getElementById("agentAllowedShellCommandsInput");
        const allowedNetworkDomainsInput = document.getElementById("agentAllowedNetworkDomainsInput");
        const approvalModeInput = document.getElementById("agentApprovalModeInput");
        const requireApprovalInput = document.getElementById("agentRequireApprovalToolsInput");
        const autoApproveInput = document.getElementById("agentAutoApproveToolsInput");
        const requireApprovalRiskLevelsInput = document.getElementById("agentRequireApprovalRiskLevelsInput");
        const requireApprovalRiskCategoriesInput = document.getElementById("agentRequireApprovalRiskCategoriesInput");
        const policyNotesInput = document.getElementById("agentPolicyNotesInput");
        const runTimeoutInput = document.getElementById("agentRunTimeoutInput");
        const runConcurrencyInput = document.getElementById("agentRunConcurrencyInput");
        const delegationModeInput = document.getElementById("agentDelegationModeInput");
        const requireDelegateWriteInput = document.getElementById("agentRequireDelegateWriteInput");
        const requireDelegateShellInput = document.getElementById("agentRequireDelegateShellInput");
        const requireDelegateMcpInput = document.getElementById("agentRequireDelegateMcpInput");
        const allowMainReadToolsInput = document.getElementById("agentAllowMainReadToolsInput");
        const verifyWorkerOutputInput = document.getElementById("agentVerifyWorkerOutputInput");
        const preferBatchDelegateInput = document.getElementById("agentPreferBatchDelegateInput");
        if (workspaceModeInput) workspaceModeInput.value = workspacePolicy.mode;
        if (allowSessionOverrideInput) allowSessionOverrideInput.checked = Boolean(workspacePolicy.allow_session_override);
        if (readableRootsInput) readableRootsInput.value = joinCommaSeparatedValues(workspacePolicy.readable_roots);
        if (writableRootsInput) writableRootsInput.value = joinCommaSeparatedValues(workspacePolicy.writable_roots);
        if (readOnlyToolsInput) readOnlyToolsInput.value = joinCommaSeparatedValues(workspacePolicy.read_only_tools);
        if (disabledToolsInput) disabledToolsInput.value = joinCommaSeparatedValues(workspacePolicy.disabled_tools);
        if (allowedShellCommandsInput) allowedShellCommandsInput.value = joinCommaSeparatedValues(workspacePolicy.allowed_shell_command_prefixes);
        if (allowedNetworkDomainsInput) allowedNetworkDomainsInput.value = joinCommaSeparatedValues(workspacePolicy.allowed_network_domains);
        if (approvalModeInput) approvalModeInput.value = approvalPolicy.mode;
        if (requireApprovalInput) requireApprovalInput.value = joinCommaSeparatedValues(approvalPolicy.require_approval_tools);
        if (autoApproveInput) autoApproveInput.value = joinCommaSeparatedValues(approvalPolicy.auto_approve_tools);
        if (requireApprovalRiskLevelsInput) requireApprovalRiskLevelsInput.value = joinCommaSeparatedValues(approvalPolicy.require_approval_risk_levels);
        if (requireApprovalRiskCategoriesInput) requireApprovalRiskCategoriesInput.value = joinCommaSeparatedValues(approvalPolicy.require_approval_risk_categories);
        if (policyNotesInput) policyNotesInput.value = approvalPolicy.notes || "";
        if (runTimeoutInput) runTimeoutInput.value = runPolicy.timeout_seconds || "";
        if (runConcurrencyInput) runConcurrencyInput.value = String(runPolicy.max_concurrent_runs || 1);
        if (delegationModeInput) delegationModeInput.value = delegationPolicy.mode;
        if (requireDelegateWriteInput) requireDelegateWriteInput.checked = Boolean(delegationPolicy.require_delegate_for_write_actions);
        if (requireDelegateShellInput) requireDelegateShellInput.checked = Boolean(delegationPolicy.require_delegate_for_shell);
        if (requireDelegateMcpInput) requireDelegateMcpInput.checked = Boolean(delegationPolicy.require_delegate_for_stateful_mcp);
        if (allowMainReadToolsInput) allowMainReadToolsInput.checked = Boolean(delegationPolicy.allow_main_agent_read_tools);
        if (verifyWorkerOutputInput) verifyWorkerOutputInput.checked = Boolean(delegationPolicy.verify_worker_output);
        if (preferBatchDelegateInput) preferBatchDelegateInput.checked = Boolean(delegationPolicy.prefer_batch_delegate);
    }

    function readAgentPolicyPayload() {
        const workspaceModeInput = document.getElementById("agentWorkspaceModeInput");
        const allowSessionOverrideInput = document.getElementById("agentAllowSessionOverrideInput");
        const readableRootsInput = document.getElementById("agentReadableRootsInput");
        const writableRootsInput = document.getElementById("agentWritableRootsInput");
        const readOnlyToolsInput = document.getElementById("agentReadOnlyToolsInput");
        const disabledToolsInput = document.getElementById("agentDisabledToolsInput");
        const allowedShellCommandsInput = document.getElementById("agentAllowedShellCommandsInput");
        const allowedNetworkDomainsInput = document.getElementById("agentAllowedNetworkDomainsInput");
        const approvalModeInput = document.getElementById("agentApprovalModeInput");
        const requireApprovalInput = document.getElementById("agentRequireApprovalToolsInput");
        const autoApproveInput = document.getElementById("agentAutoApproveToolsInput");
        const requireApprovalRiskLevelsInput = document.getElementById("agentRequireApprovalRiskLevelsInput");
        const requireApprovalRiskCategoriesInput = document.getElementById("agentRequireApprovalRiskCategoriesInput");
        const policyNotesInput = document.getElementById("agentPolicyNotesInput");
        const runTimeoutInput = document.getElementById("agentRunTimeoutInput");
        const runConcurrencyInput = document.getElementById("agentRunConcurrencyInput");
        const delegationModeInput = document.getElementById("agentDelegationModeInput");
        const requireDelegateWriteInput = document.getElementById("agentRequireDelegateWriteInput");
        const requireDelegateShellInput = document.getElementById("agentRequireDelegateShellInput");
        const requireDelegateMcpInput = document.getElementById("agentRequireDelegateMcpInput");
        const allowMainReadToolsInput = document.getElementById("agentAllowMainReadToolsInput");
        const verifyWorkerOutputInput = document.getElementById("agentVerifyWorkerOutputInput");
        const preferBatchDelegateInput = document.getElementById("agentPreferBatchDelegateInput");
        const timeoutValue = Number.parseInt(runTimeoutInput?.value || "", 10);
        const concurrencyValue = Number.parseInt(runConcurrencyInput?.value || "", 10);
        const workspaceType = workspaceModeInput?.value === "shared" ? "shared" : "isolated";

        return {
            workspace_type: workspaceType,
            workspace_policy: {
                mode: workspaceType,
                allow_session_override: Boolean(allowSessionOverrideInput?.checked),
                readable_roots: splitCommaSeparatedValues(readableRootsInput?.value),
                writable_roots: splitCommaSeparatedValues(writableRootsInput?.value),
                read_only_tools: splitCommaSeparatedValues(readOnlyToolsInput?.value),
                disabled_tools: splitCommaSeparatedValues(disabledToolsInput?.value),
                allowed_shell_command_prefixes: splitCommaSeparatedValues(allowedShellCommandsInput?.value),
                allowed_network_domains: splitCommaSeparatedValues(allowedNetworkDomainsInput?.value),
            },
            approval_policy: {
                mode: approvalModeInput?.value || "default",
                require_approval_tools: splitCommaSeparatedValues(requireApprovalInput?.value),
                auto_approve_tools: splitCommaSeparatedValues(autoApproveInput?.value),
                require_approval_risk_levels: splitCommaSeparatedValues(requireApprovalRiskLevelsInput?.value),
                require_approval_risk_categories: splitCommaSeparatedValues(requireApprovalRiskCategoriesInput?.value),
                notes: policyNotesInput?.value.trim() || "",
            },
            run_policy: {
                timeout_seconds: Number.isFinite(timeoutValue) && timeoutValue > 0 ? timeoutValue : null,
                max_concurrent_runs: Number.isFinite(concurrencyValue) && concurrencyValue > 0 ? concurrencyValue : 1,
            },
            delegation_policy: {
                mode: delegationModeInput?.value || "prefer_delegate",
                require_delegate_for_write_actions: Boolean(requireDelegateWriteInput?.checked),
                require_delegate_for_shell: Boolean(requireDelegateShellInput?.checked),
                require_delegate_for_stateful_mcp: Boolean(requireDelegateMcpInput?.checked),
                allow_main_agent_read_tools: Boolean(allowMainReadToolsInput?.checked),
                verify_worker_output: Boolean(verifyWorkerOutputInput?.checked),
                prefer_batch_delegate: Boolean(preferBatchDelegateInput?.checked),
            },
        };
    }

    function renderAgentPolicyBadges(agent) {
        const { tools, workspacePolicy, approvalPolicy, runPolicy, delegationPolicy } = normalizeTemplatePolicies(agent);
        const modeMeta = getDelegationModeMeta(delegationPolicy.mode);
        const badges = [
            { label: `Tools ${tools.length || 0}`, tone: "border-primary/20 bg-primary/10 text-primary" },
            { label: `Workspace ${workspacePolicy.mode}`, tone: "border-secondary/20 bg-secondary/10 text-secondary" },
            { label: `Risk ${approvalPolicy.mode}`, tone: "border-amber-400/20 bg-amber-400/10 text-amber-200" },
            { label: modeMeta.label, tone: modeMeta.tone },
            {
                label: runPolicy.timeout_seconds ? `Timeout ${runPolicy.timeout_seconds}s` : "Timeout inherited",
                tone: "border-white/10 bg-white/5 text-on-surface-variant",
            },
            {
                label: `Concurrency ${runPolicy.max_concurrent_runs || 1}`,
                tone: "border-emerald-400/20 bg-emerald-400/10 text-emerald-300",
            },
        ];

        return badges.map((badge) => `
            <span class="inline-flex items-center rounded-full border px-2.5 py-1 text-[10px] font-mono uppercase tracking-[0.18em] ${badge.tone}">
                ${escapeHtml(badge.label)}
            </span>
        `).join("");
    }

    function renderInstalledSkills(skills) {
        const container = document.getElementById("installedSkillsContainer");
        if (!container) return;

        const installed = Array.isArray(skills) ? skills : [];
        const canManage = Boolean(agentModalState.agentId)
            && !agentModalState.isSystem
            && !isSkillInstallActive(agentModalState.skillInstallStatus);
        if (installed.length === 0) {
            container.innerHTML = `<div class="text-sm text-on-surface-variant">No skills installed yet.</div>`;
            return;
        }

        container.innerHTML = installed.map((skill) => `
            <div class="flex items-start justify-between gap-3 p-3 rounded-xl bg-surface-container-highest border border-outline-variant/10">
                <div>
                    <div class="text-sm font-bold text-on-surface">${escapeHtml(skill.name || "")}</div>
                    <div class="text-xs text-on-surface-variant mt-1">${escapeHtml(skill.description || "")}</div>
                </div>
                <div class="flex items-center gap-2 shrink-0">
                    <span class="text-[10px] uppercase tracking-widest text-primary">Installed</span>
                    ${canManage ? `
                        <button
                            onclick="window.deleteAgentSkill('${escapeHtml(skill.name || "").replace(/'/g, "\\'")}')"
                            class="p-1.5 rounded-lg border border-error/20 bg-error/10 text-error/80 hover:text-error hover:bg-error/20 transition-colors"
                            title="Delete skill"
                        >
                            <span class="material-symbols-outlined text-[14px]">delete</span>
                        </button>
                    ` : ""}
                </div>
            </div>
        `).join("");
    }

    function createSkillSearchGroup({ capabilityName, keyword, reason, candidates, error }) {
        return {
            capability_name: capabilityName || "Recommended Capability",
            keyword: keyword || "",
            reason: reason || "",
            skill_candidates: Array.isArray(candidates) ? candidates : [],
            skill_search_error: error || "",
        };
    }

    function setActiveSkillGroup(index) {
        const groups = Array.isArray(agentModalState.searchResults) ? agentModalState.searchResults : [];
        if (!groups.length) {
            agentModalState.activeSkillGroupIndex = 0;
            return;
        }

        const normalizedIndex = Number.isInteger(index) ? index : 0;
        agentModalState.activeSkillGroupIndex = Math.max(0, Math.min(normalizedIndex, groups.length - 1));
    }

    function renderSkillSearchResults(groups, selectedPackages = []) {
        const container = document.getElementById("skillSearchResults");
        if (!container) return;

        const selected = new Set(Array.isArray(selectedPackages) ? selectedPackages : []);
        const items = Array.isArray(groups) ? groups : [];
        const installedNames = new Set((agentModalState.installedSkills || []).map((skill) => skill.name));
        if (items.length === 0) {
            container.innerHTML = `<div class="text-sm text-on-surface-variant">Search skills by keyword or use Auto Brainstorm to get suggestions.</div>`;
            return;
        }

        setActiveSkillGroup(agentModalState.activeSkillGroupIndex);
        const activeGroup = items[agentModalState.activeSkillGroupIndex] || items[0];
        const capabilityName = activeGroup.capability_name || activeGroup.name || "Recommended Capability";
        const keyword = activeGroup.keyword || "";
        const reason = activeGroup.reason || "";
        const error = activeGroup.skill_search_error || activeGroup.error || "";
        const candidates = Array.isArray(activeGroup.skill_candidates)
            ? activeGroup.skill_candidates
            : Array.isArray(activeGroup.candidates)
                ? activeGroup.candidates
                : [];

        const tabsHtml = items.map((group, index) => {
            const groupCandidates = Array.isArray(group.skill_candidates)
                ? group.skill_candidates
                : Array.isArray(group.candidates)
                    ? group.candidates
                    : [];
            const label = group.keyword || group.capability_name || group.name || `Group ${index + 1}`;
            const isActive = index === agentModalState.activeSkillGroupIndex;
            const selectedCount = groupCandidates.filter((candidate) => selected.has(candidate.package_name || "")).length;
            return `
                <button
                    type="button"
                    onclick="window.selectSkillSearchGroup(${index})"
                    class="inline-flex items-center gap-2 px-3 py-2 rounded-xl border text-xs font-bold transition-colors ${isActive ? "bg-primary/10 border-primary/40 text-primary" : "bg-surface-container-highest border-outline-variant/10 text-on-surface-variant hover:border-primary/30 hover:text-on-surface"}"
                >
                    <span>${escapeHtml(label)}</span>
                    <span class="text-[10px] uppercase tracking-[0.18em] ${isActive ? "text-primary/80" : "text-on-surface-variant/70"}">${groupCandidates.length}</span>
                    ${selectedCount > 0 ? `<span class="inline-flex items-center justify-center min-w-5 h-5 px-1 rounded-full bg-emerald-400/15 text-emerald-400 text-[10px]">${selectedCount}</span>` : ""}
                </button>
            `;
        }).join("");

        const candidatesHtml = candidates.length > 0
            ? candidates.map((item) => {
                const pkg = item.package_name || "";
                const checked = selected.has(pkg) ? "checked" : "";
                const meta = [item.version, item.score ? `score ${item.score}` : ""].filter(Boolean).join(" | ");
                const installed = installedNames.has(pkg);
                return `
                    <label class="flex items-start gap-3 p-3 bg-surface-container-highest rounded-xl border border-outline-variant/10 hover:border-primary/30 transition-colors cursor-pointer">
                        <input
                            type="checkbox"
                            class="skill-package-checkbox mt-1 rounded border-outline-variant/30 bg-surface-container"
                            value="${escapeHtml(pkg)}"
                            ${checked}
                            ${installed ? "disabled" : ""}
                            onchange="window.toggleSkillPackageSelection('${escapeHtml(pkg).replace(/'/g, "\\'")}', this.checked)"
                        >
                        <div class="flex-1 min-w-0">
                            <div class="flex items-center justify-between gap-3">
                                <div class="text-sm font-bold text-on-surface">${escapeHtml(pkg)}</div>
                                ${installed ? `<span class="text-[10px] uppercase tracking-widest text-emerald-400">Installed</span>` : ""}
                            </div>
                            <div class="text-xs text-on-surface-variant mt-1">${escapeHtml(item.description || "")}</div>
                            <div class="text-[10px] uppercase tracking-widest text-primary/70 mt-2">${escapeHtml(meta)}</div>
                        </div>
                    </label>
                `;
            }).join("")
            : `<div class="text-sm text-on-surface-variant">${escapeHtml(error || "No skill candidates found for this capability yet.")}</div>`;

        container.innerHTML = `
            <section class="rounded-xl border border-outline-variant/10 bg-surface-container p-4 space-y-4">
                <div class="flex flex-wrap gap-2">
                    ${tabsHtml}
                </div>
                <div class="rounded-xl border border-outline-variant/10 bg-surface-container-low p-4 space-y-3">
                    <div class="flex flex-wrap items-start justify-between gap-3">
                        <div>
                            <div class="text-sm font-bold text-on-surface">${escapeHtml(capabilityName)}</div>
                            ${reason ? `<div class="text-xs text-on-surface-variant mt-1">${escapeHtml(reason)}</div>` : ""}
                        </div>
                        ${keyword ? `<span class="inline-flex items-center px-2.5 py-1 rounded-full border border-primary/20 bg-primary/10 text-[10px] font-bold uppercase tracking-[0.18em] text-primary">${escapeHtml(keyword)}</span>` : ""}
                    </div>
                    <div class="grid grid-cols-1 gap-3">
                        ${candidatesHtml}
                    </div>
                </div>
            </section>
        `;
    }

    function clearAgentModalPoll() {
        if (!agentModalPollTimer) return;
        clearTimeout(agentModalPollTimer);
        agentModalPollTimer = null;
    }

    function setAgentModalStateFromAgent(agent) {
        agentModalState.agentId = agent?.id || "";
        agentModalState.isSystem = Boolean(agent?.is_system);
        agentModalState.installedSkills = Array.isArray(agent?.skills) ? agent.skills : [];
        agentModalState.skillInstallStatus = agent?.skill_install_status || null;
    }

    function updateAgentSkillActionControls() {
        const installBtn = document.getElementById("btnInstallSelectedSkills");
        const searchHint = document.getElementById("skillSearchHint");
        const brainstormBtn = document.getElementById("btnBrainstormAgent");
        const searchBtn = document.getElementById("btnSearchSkills");
        const isExistingAgent = Boolean(agentModalState.agentId);
        const isReadonly = Boolean(agentModalState.isSystem);
        const installActive = isSkillInstallActive(agentModalState.skillInstallStatus);
        const selectedCount = agentModalState.selectedPackages.size;

        if (searchHint) {
            if (isReadonly) {
                searchHint.textContent = "系统模板仅展示当前技能清单与推荐能力维度，不支持在这里安装或删除技能。";
            } else {
                searchHint.textContent = isExistingAgent
                    ? "Search skills manually, then install or remove them without leaving this configuration form."
                    : "Search skills manually. Selected skills will be installed after you save the new agent.";
            }
        }

        if (brainstormBtn) {
            brainstormBtn.disabled = isReadonly;
            brainstormBtn.classList.toggle("opacity-50", brainstormBtn.disabled);
            brainstormBtn.classList.toggle("cursor-not-allowed", brainstormBtn.disabled);
        }

        if (searchBtn) {
            searchBtn.disabled = isReadonly;
            searchBtn.classList.toggle("opacity-50", searchBtn.disabled);
            searchBtn.classList.toggle("cursor-not-allowed", searchBtn.disabled);
        }

        if (installBtn) {
            installBtn.disabled = isReadonly || !isExistingAgent || installActive || selectedCount === 0;
            installBtn.classList.toggle("opacity-50", installBtn.disabled);
            installBtn.classList.toggle("cursor-not-allowed", installBtn.disabled);
            installBtn.innerHTML = `
                <span class="material-symbols-outlined text-[18px]">download</span>
                ${isReadonly
                    ? "只读展示"
                    : isExistingAgent
                        ? `Install Selected${selectedCount > 0 ? ` (${selectedCount})` : ""}`
                        : "Install After Save"}
            `;
        }
    }

    function syncAgentModalSkillViews() {
        renderInstalledSkills(agentModalState.installedSkills);
        renderSkillInstallStatus(agentModalState.skillInstallStatus, (agentModalState.installedSkills || []).length);
        renderSkillSearchResults(agentModalState.searchResults, Array.from(agentModalState.selectedPackages));
        updateAgentSkillActionControls();
    }

    async function refreshAgentModalSkillState(agentId, options = {}) {
        const { scheduleNextPoll = false } = options;
        if (!agentId) return;

        try {
            const agent = await fetchJson(`/api/agents/${agentId}`);
            if (agentId !== agentModalState.agentId) {
                return;
            }
            setAgentModalStateFromAgent(agent);
            syncAgentModalSkillViews();
            await window.renderAgentsGrid({ background: true });
            if (scheduleNextPoll && isSkillInstallActive(agent.skill_install_status)) {
                agentModalPollTimer = setTimeout(() => {
                    refreshAgentModalSkillState(agentId, { scheduleNextPoll: true });
                }, 2500);
            }
        } catch (error) {
            console.error("Failed to refresh modal agent skill state:", error);
        }
    }

    window.toggleSkillPackageSelection = function (packageName, checked) {
        if (!packageName) return;
        if (checked) {
            agentModalState.selectedPackages.add(packageName);
        } else {
            agentModalState.selectedPackages.delete(packageName);
        }
        syncAgentModalSkillViews();
    };

    window.selectSkillSearchGroup = function (index) {
        setActiveSkillGroup(index);
        syncAgentModalSkillViews();
    };

    window.closeAgentModal = function () {
        const modal = document.getElementById("agentConfigModal");
        if (modal) modal.classList.add("hidden");
        clearAgentModalPoll();
        agentModalState.agentId = "";
        agentModalState.isSystem = false;
        agentModalState.searchResults = [];
        agentModalState.selectedPackages = new Set();
        agentModalState.installedSkills = [];
        agentModalState.skillInstallStatus = null;
        agentModalState.activeSkillGroupIndex = 0;
    };

    window.openAgentModal = async function (agentId = null) {
        const modal = document.getElementById("agentConfigModal");
        if (!modal) return;

        // Setup state
        const idInput = document.getElementById("agentIdInput");
        const nameInput = document.getElementById("agentNameInput");
        const descInput = document.getElementById("agentDescInput");
        const promptInput = document.getElementById("agentPromptInput");
        const title = document.getElementById("agentModalTitle");
        const keywordLabel = document.getElementById("recommendedSkillKeyword");
        const keywordInput = document.getElementById("skillSearchKeywordInput");
        const modalSubtitle = document.getElementById("agentModalSubtitle");

        idInput.value = agentId || "";
        title.textContent = agentId ? "Edit Template Config" : "Create New Template";
        if (modalSubtitle) {
            modalSubtitle.textContent = "Fill at least one field and auto-generate the rest.";
        }
        clearAgentModalPoll();
        agentModalState.searchResults = [];
        agentModalState.selectedPackages = new Set();
        agentModalState.activeSkillGroupIndex = 0;

        let editingAgent = null;
        if (agentId) {
            try {
                editingAgent = await fetchJson(`/api/agents/${agentId}`);
            } catch (err) {
                console.error("Failed to fetch agent", err);
            }
        }

        if (editingAgent) {
            nameInput.value = editingAgent.name || "";
            descInput.value = editingAgent.description || "";
            promptInput.value = editingAgent.system_prompt || "";
            setAgentModalStateFromAgent(editingAgent);
            populateAgentPolicyFields(editingAgent);
        } else {
            nameInput.value = "";
            descInput.value = "";
            promptInput.value = "";
            setAgentModalStateFromAgent(null);
            const defaultTemplate = marketplaceAgentCatalog.get(DEFAULT_AGENT_ID) || null;
            populateAgentPolicyFields(
                defaultTemplate ? { ...defaultTemplate, is_system: false } : null
            );
        }

        if (editingAgent?.is_system) {
            title.textContent = "查看系统模板";
            if (modalSubtitle) {
                modalSubtitle.textContent = "系统模板仅用于查看默认 supervisor 策略与工具边界。主 / 子 agent 的 API 组请在 API Settings 中配置。";
            }
        }

        keywordLabel.textContent = "Run Auto Brainstorm to get Chinese-first suggestions";
        if (keywordInput) keywordInput.value = "";
        setAgentModalReadonlyState(Boolean(editingAgent?.is_system));
        syncAgentModalSkillViews();

        modal.classList.remove("hidden");
        if (editingAgent && isSkillInstallActive(editingAgent.skill_install_status)) {
            agentModalPollTimer = setTimeout(() => {
                refreshAgentModalSkillState(editingAgent.id, { scheduleNextPoll: true });
            }, 2500);
        }
    };

    window.brainstormAgentConfig = async function () {
        if (agentModalState.isSystem) {
            alert("系统模板为只读展示，不能在这里自动生成配置。");
            return;
        }
        const nameInput = document.getElementById("agentNameInput");
        const descInput = document.getElementById("agentDescInput");
        const promptInput = document.getElementById("agentPromptInput");
        const keywordLabel = document.getElementById("recommendedSkillKeyword");
        const keywordInput = document.getElementById("skillSearchKeywordInput");
        const brainstormBtn = document.getElementById("btnBrainstormAgent");
        const icon = brainstormBtn.querySelector("span");

        const payload = {
            name: nameInput.value.trim(),
            description: descInput.value.trim(),
            system_prompt: promptInput.value.trim()
        };

        if (!payload.name && !payload.description && !payload.system_prompt) {
            alert("Please fill in at least one field so the AI has context to brainstorm from.");
            return;
        }

        brainstormBtn.disabled = true;
        brainstormBtn.classList.add("opacity-50", "cursor-not-allowed");
        icon.textContent = "progress_activity";
        icon.classList.add("animate-spin");
        icon.classList.remove("animate-pulse");

        try {
            const data = await fetchJson("/api/agents/brainstorm", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (data.name) nameInput.value = data.name;
            if (data.description) descInput.value = data.description;
            if (data.system_prompt) promptInput.value = data.system_prompt;
            const capabilityGroups = Array.isArray(data.skill_capability_groups) ? data.skill_capability_groups : [];
            const capabilityNames = capabilityGroups.map((group) => group.capability_name).filter(Boolean);
            keywordLabel.textContent = capabilityNames.length
                ? capabilityNames.join(" / ")
                : data.recommended_skill_keyword || "No capability dimensions returned";
            if (keywordInput) keywordInput.value = data.recommended_skill_keyword || "";
            agentModalState.selectedPackages = new Set();
            agentModalState.searchResults = capabilityGroups;
            agentModalState.activeSkillGroupIndex = 0;
            syncAgentModalSkillViews();
            const failedGroups = capabilityGroups.filter((group) => group.skill_search_error);
            if (failedGroups.length > 0) {
                alert("Some capability searches failed: " + failedGroups.map((group) => `${group.capability_name}: ${group.skill_search_error}`).join(" | "));
            }

        } catch (err) {
            alert("Failed to Auto Brainstorm: " + err.message);
        } finally {
            brainstormBtn.disabled = false;
            brainstormBtn.classList.remove("opacity-50", "cursor-not-allowed");
            icon.textContent = "magic_button";
            icon.classList.remove("animate-spin");
            // Do not re-add pulse unless we want it continuously pulsing.
        }
    };

    window.searchAgentSkills = async function () {
        if (agentModalState.isSystem) {
            alert("系统模板为只读展示，不能在这里检索或安装技能。");
            return;
        }
        const keywordInput = document.getElementById("skillSearchKeywordInput");
        const keywordLabel = document.getElementById("recommendedSkillKeyword");
        const searchBtn = document.getElementById("btnSearchSkills");
        const keyword = keywordInput?.value.trim() || "";

        if (!keyword) {
            alert("Please enter a skill keyword to search.");
            return;
        }

        searchBtn.disabled = true;
        searchBtn.classList.add("opacity-50", "cursor-not-allowed");

        try {
            const results = await fetchJson(`/api/skills/search?keyword=${encodeURIComponent(keyword)}`);
            agentModalState.selectedPackages = new Set();
            agentModalState.searchResults = [
                createSkillSearchGroup({
                    capabilityName: "Manual Search",
                    keyword,
                    reason: "Results from your direct keyword search.",
                    candidates: Array.isArray(results) ? results : [],
                })
            ];
            agentModalState.activeSkillGroupIndex = 0;
            keywordLabel.textContent = keyword;
            syncAgentModalSkillViews();
        } catch (err) {
            alert("Failed to search skills: " + err.message);
        } finally {
            searchBtn.disabled = false;
            searchBtn.classList.remove("opacity-50", "cursor-not-allowed");
        }
    };

    window.installSelectedSkills = async function () {
        if (agentModalState.isSystem) {
            alert("系统模板为只读展示，不能在这里安装技能。");
            return;
        }
        const agentId = agentModalState.agentId;
        const packages = Array.from(agentModalState.selectedPackages);

        if (!agentId) {
            alert("Please save this agent first, then install skills directly.");
            return;
        }
        if (!packages.length) {
            alert("Please select at least one skill package.");
            return;
        }

        try {
            const updatedAgent = await fetchJson(`/api/agents/${agentId}/skills/install`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ package_names: packages })
            });
            setAgentModalStateFromAgent(updatedAgent);
            agentModalState.selectedPackages = new Set();
            syncAgentModalSkillViews();
            await window.renderAgentsGrid({ background: true });
            if (isSkillInstallActive(updatedAgent.skill_install_status)) {
                clearAgentGridPoll();
                scheduleAgentGridPoll();
                agentModalPollTimer = setTimeout(() => {
                    refreshAgentModalSkillState(agentId, { scheduleNextPoll: true });
                }, 2500);
            }
        } catch (err) {
            alert("Failed to install selected skills: " + err.message);
        }
    };

    window.deleteAgentSkill = async function (skillName) {
        const agentId = agentModalState.agentId;
        if (agentModalState.isSystem) {
            alert("系统模板为只读展示，不能在这里删除技能。");
            return;
        }
        if (!agentId || !skillName) return;
        if (!confirm(`Delete installed skill "${skillName}" from this agent?`)) return;

        try {
            const updatedAgent = await fetchJson(`/api/agents/${agentId}/skills/${encodeURIComponent(skillName)}`, {
                method: "DELETE"
            });
            setAgentModalStateFromAgent(updatedAgent);
            syncAgentModalSkillViews();
            await window.renderAgentsGrid({ background: true });
        } catch (err) {
            alert("Failed to delete skill: " + err.message);
        }
    };

    window.saveAgentConfig = async function () {
        if (agentModalState.isSystem) {
            alert("系统模板为只读展示，不能直接保存修改。");
            return;
        }
        const id = document.getElementById("agentIdInput").value;
        const name = document.getElementById("agentNameInput").value.trim();
        const description = document.getElementById("agentDescInput").value.trim();
        const system_prompt = document.getElementById("agentPromptInput").value.trim();
        const spinner = document.getElementById("saveAgentSpinner");

        if (!name || !system_prompt) {
            alert("Agent Name and System Prompt are required.");
            return;
        }

        const selectedSkillPackages = Array.from(agentModalState.selectedPackages);
        const policyPayload = readAgentPolicyPayload();

        const payload = {
            name,
            description,
            system_prompt,
            selected_skill_packages: selectedSkillPackages,
            mcp_configs: [],
            ...policyPayload,
        };

        spinner.classList.remove("hidden");
        try {
            let savedAgent;
            if (id) {
                savedAgent = await fetchJson(`/api/agents/${id}`, {
                    method: "PUT",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
            } else {
                savedAgent = await fetchJson(`/api/agents`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
            }
            window.closeAgentModal();
            await window.renderAgentsGrid({ background: true });
            if (savedAgent && isSkillInstallActive(savedAgent.skill_install_status)) {
                scheduleAgentGridPoll();
            }
        } catch (err) {
            alert("Failed to save agent: " + err.message);
        } finally {
            spinner.classList.add("hidden");
        }
    };

    window.deleteCustomAgent = async function (e, id) {
        e.stopPropagation();
        if (!confirm("Are you sure you want to delete this agent configuration?")) return;

        try {
            const response = await fetch(`/api/agents/${id}`, { method: "DELETE" });
            if (!response.ok) throw new Error(`HTTP Error ${response.status}`);
            window.renderAgentsGrid();
        } catch (err) {
            alert("Failed to delete agent: " + err.message);
        }
    };

    window.deployCustomAgent = async function (e, id) {
        e.stopPropagation();
        try {
            await createSession({ openChat: true, agent_id: id });
        } catch (err) {
            alert("Failed to deploy agent: " + err.message);
        }
    };

    window.renderAgentsGrid = async function (options = {}) {
        const grid = document.getElementById("agentsGrid");
        if (!grid) return;
        clearAgentGridPoll();

        const background = Boolean(options.background);
        const hasExistingContent = grid.children.length > 0;
        if (!background || !hasExistingContent) {
            grid.innerHTML = '<div class="col-span-full flex justify-center py-10"><span class="material-symbols-outlined animate-spin text-4xl text-primary">progress_activity</span></div>';
        }

        try {
            const agents = await fetchJson("/api/agents");
            applyAgentCatalog(agents);
            const fragment = document.createDocumentFragment();
            const hasActiveInstall = agents.some(agent => isSkillInstallActive(agent.skill_install_status));

            agents.forEach(agent => {
                const isSystem = agent.is_system;
                const installStatus = normalizeSkillInstallStatus(agent.skill_install_status);
                const { delegationPolicy } = normalizeTemplatePolicies(agent);
                const delegationMeta = getDelegationModeMeta(delegationPolicy.mode);
                const card = document.createElement("div");
                card.className = "bg-surface-container-low border border-outline-variant/20 p-6 rounded-2xl hover:border-primary/50 hover:bg-surface-container-high transition-all shadow-lg group relative flex flex-col h-full cursor-pointer";
                card.onclick = () => window.openAgentModal(agent.id);

                let badgeHtml = isSystem
                    ? `<div class="absolute top-4 right-4 flex items-center gap-1 bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border border-emerald-500/20">
                           <span class="material-symbols-outlined text-[12px]">verified</span> Official
                       </div>`
                    : `<div class="absolute top-4 right-4 flex items-center gap-1 z-10">
                           <button onclick="window.deleteCustomAgent(event, '${agent.id}')" class="text-error/70 hover:text-error bg-error/10 hover:bg-error/20 p-1.5 rounded-lg transition-colors border border-error/10" title="Delete Agent">
                               <span class="material-symbols-outlined text-[14px]">delete</span>
                           </button>
                       </div>`;

                let iconHtml = isSystem ? `<span class="material-symbols-outlined text-[24px]">code</span>` : `<span class="material-symbols-outlined text-[24px]">smart_toy</span>`;
                let iconColor = isSystem ? "text-primary bg-primary/10 border-primary/20" : "text-[#d575ff] bg-[#d575ff]/10 border-[#d575ff]/20";
                const policyBadges = renderAgentPolicyBadges(agent);

                card.innerHTML = `
                    ${badgeHtml}
                    <div class="flex items-center gap-4 mb-4 mt-2">
                        <div class="w-12 h-12 ${iconColor} rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform border">
                            ${iconHtml}
                        </div>
                        <div>
                            <h3 class="font-bold text-base font-headline">${escapeHtml(agent.name)}</h3>
                            <p class="text-[11px] text-on-surface-variant font-mono text-primary/80">Template v${escapeHtml(String(agent.version || 1))}</p>
                            <div class="mt-2">${renderSkillStatusSummary(installStatus, (agent.skills || []).length, { compact: true })}</div>
                        </div>
                    </div>
                    <p class="text-sm text-on-surface-variant mb-4 line-clamp-3 leading-relaxed">${escapeHtml(agent.description) || "No description provided."}</p>
                    <div class="mb-4 rounded-xl border border-outline-variant/10 bg-surface-container p-4">
                        <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">Template Configuration</div>
                        <div class="mt-3 flex flex-wrap gap-2">
                            ${policyBadges}
                        </div>
                        <div class="mt-3 text-[11px] leading-relaxed text-on-surface-variant">
                            ${escapeHtml(delegationMeta.summary)}
                        </div>
                        <div class="mt-3 text-[11px] leading-relaxed text-on-surface-variant">
                            ${escapeHtml(buildRoutingSummary())}
                        </div>
                    </div>
                    <button onclick="window.deployCustomAgent(event, '${agent.id}')" class="w-full py-2.5 bg-white/5 border border-white/10 hover:bg-primary/20 hover:text-primary hover:border-primary/50 font-bold rounded-xl text-xs transition-colors tracking-wide uppercase flex items-center justify-center gap-2">
                        <span class="material-symbols-outlined text-[16px]">play_arrow</span> Deploy to Workspace
                    </button>
                `;
                fragment.appendChild(card);
            });

            if (hasActiveInstall) {
                scheduleAgentGridPoll();
            }

            const addCard = document.createElement("div");
            addCard.className = "bg-surface-container-low border border-outline-variant/20 p-6 rounded-2xl hover:border-primary/50 flex flex-col justify-center items-center cursor-pointer border-dashed hover:bg-surface-container-high transition-all group";
            addCard.style.minHeight = "240px";
            addCard.onclick = () => window.openAgentModal(null);
            addCard.innerHTML = `
                <div class="w-12 h-12 bg-surface-container-highest rounded-full flex items-center justify-center mb-3 group-hover:bg-primary/10 group-hover:text-primary transition-colors border border-outline-variant/10">
                    <span class="material-symbols-outlined text-[24px]">add</span>
                </div>
                <span class="text-sm font-bold text-on-surface-variant group-hover:text-primary transition-colors">Create Template</span>
            `;
            fragment.appendChild(addCard);
            grid.replaceChildren(fragment);

        } catch (err) {
            if (background && hasExistingContent) {
                console.error("Failed to refresh agents:", err);
                return;
            }
            grid.innerHTML = `<div class="col-span-full text-error p-4 flex flex-col items-center justify-center gap-2">
                <span class="material-symbols-outlined text-4xl">warning</span>
                Failed to load agents: ${err.message}
            </div>`;
        }
    };

    async function openSessionPage() {
        if (sessionId) {
            await switchSession(sessionId);
            return;
        }

        if (sessions.length) {
            await switchSession(sessions[0].session_id);
            return;
        }

        await createSession({ openChat: true });
    }

    function renderStructuredUserMessage(content) {
        if (typeof content === "string") {
            return `<p class="text-sm text-on-surface leading-relaxed">${renderPlainText(content)}</p>`;
        }

        const blocks = normalizeMessageBlocks(content);
        if (!blocks.length) {
            return `<p class="text-sm text-on-surface leading-relaxed">${renderPlainText(structuredMessageSummary(content))}</p>`;
        }

        const textSections = [];
        const uploadCards = [];
        const artifactCards = [];

        blocks.forEach((block) => {
            if (block.type === "text" && typeof block.text === "string" && block.text.trim()) {
                textSections.push(
                    `<p class="text-sm text-on-surface leading-relaxed">${renderPlainText(block.text)}</p>`
                );
                return;
            }

            if (block.type === "uploaded_file") {
                const displayName = escapeHtml(block.original_name || block.safe_name || block.upload_id || "未命名文件");
                const metaParts = [];
                if (typeof block.mime_type === "string" && block.mime_type.trim()) {
                    metaParts.push(escapeHtml(block.mime_type.trim()));
                }
                const sizeLabel = formatFileSize(Number(block.size_bytes));
                if (sizeLabel) {
                    metaParts.push(escapeHtml(sizeLabel));
                }
                const relativePath = typeof block.relative_path === "string" ? block.relative_path.trim() : "";
                uploadCards.push(`
                    <div class="rounded-xl border border-outline-variant/15 bg-surface-container px-4 py-3 shadow-sm">
                        <div class="flex items-center gap-2 text-[11px] font-semibold tracking-[0.18em] text-on-surface-variant uppercase">
                            <span class="material-symbols-outlined text-sm">attach_file</span>
                            <span>已附带文件</span>
                        </div>
                        <div class="mt-2 text-sm font-medium text-on-surface">${displayName}</div>
                        ${metaParts.length ? `<div class="mt-1 text-xs text-on-surface-variant">${metaParts.join(" · ")}</div>` : ""}
                        ${relativePath ? `<div class="mt-2 text-[11px] font-mono text-on-surface-variant break-all">工作区路径：${escapeHtml(relativePath)}</div>` : ""}
                    </div>
                `);
                return;
            }

            if (block.type === "artifact_ref") {
                const displayName = escapeHtml(block.display_name || block.uri || block.artifact_id || "未命名产物");
                const summary = typeof block.summary === "string" ? block.summary.trim() : "";
                const uri = typeof block.uri === "string" ? block.uri.trim() : "";
                artifactCards.push(`
                    <div class="rounded-xl border border-outline-variant/15 bg-surface-container px-4 py-3 shadow-sm">
                        <div class="flex items-center gap-2 text-[11px] font-semibold tracking-[0.18em] text-on-surface-variant uppercase">
                            <span class="material-symbols-outlined text-sm">inventory_2</span>
                            <span>引用产物</span>
                        </div>
                        <div class="mt-2 text-sm font-medium text-on-surface">${displayName}</div>
                        ${summary ? `<div class="mt-1 text-xs text-on-surface-variant">${escapeHtml(summary)}</div>` : ""}
                        ${uri ? `<div class="mt-2 text-[11px] font-mono text-on-surface-variant break-all">URI：${escapeHtml(uri)}</div>` : ""}
                    </div>
                `);
            }
        });

        const sections = [];
        if (textSections.length) {
            sections.push(`<div class="space-y-3">${textSections.join("")}</div>`);
        }
        if (uploadCards.length) {
            sections.push(`<div class="mt-3 flex flex-col gap-2">${uploadCards.join("")}</div>`);
        }
        if (artifactCards.length) {
            sections.push(`<div class="mt-3 flex flex-col gap-2">${artifactCards.join("")}</div>`);
        }

        return sections.join("") || `<p class="text-sm text-on-surface leading-relaxed">${renderPlainText(structuredMessageSummary(content))}</p>`;
    }

    function appendMessage(role, content) {
        const el = document.createElement("div");
        el.className = `flex flex-col items-end gap-2 max-w-3xl ml-auto message ${role} mb-4`;
        el.innerHTML = `
            <div class="bg-surface-container-highest px-6 py-4 rounded-2xl rounded-tr-none border border-outline-variant/10 shadow-xl">
                ${role === "user" ? renderStructuredUserMessage(content) : `<p class="text-sm text-on-surface leading-relaxed">${renderPlainText(messageText(content))}</p>`}
            </div>
            <span class="text-[10px] font-mono text-on-surface-variant uppercase mr-2">${role === 'user' ? 'User' : 'System'} • ${formatTimestamp(new Date())}</span>
        `;
        $messages.appendChild(el);
        scrollToBottom(true);
    }

    function appendAssistantContainer(withDots = true) {
        const agentDisplayName = escapeHtml(getAgentDisplayName());
        const el = document.createElement("div");
        el.className = `flex flex-col items-start gap-4 max-w-4xl message assistant mb-8`;
        el.innerHTML = `
            <div class="flex items-center gap-3">
                <div class="w-8 h-8 rounded-lg bg-secondary flex items-center justify-center shrink-0">
                    <span class="material-symbols-outlined text-white text-lg">smart_toy</span>
                </div>
                <span class="text-xs font-bold font-headline tracking-wider pt-1">${agentDisplayName}</span>
                <span class="text-[10px] font-mono text-on-surface-variant pt-1">${formatTimestamp(new Date()).split(' ')[1] || ''}</span>
            </div>
            <div class="w-full stream">
                ${withDots ? '<div class="dots flex h-6 items-center px-4"><span></span><span></span><span></span></div>' : ""}
            </div>
        `;
        $messages.appendChild(el);
        scrollToBottom(true);
        const stream = el.querySelector(".stream");
        if (stream) {
            conversationAssistantTurns.push(stream);
        }
        return stream;
    }

    function setAssistantStreamRunId(container, runId) {
        if (!container) {
            return;
        }
        const normalizedRunId = String(runId || "").trim();
        if (!normalizedRunId) {
            return;
        }
        container.dataset.assistantRunId = normalizedRunId;
        const rootRunId = resolveRootRunId(normalizedRunId);
        if (rootRunId) {
            container.dataset.assistantRootRunId = rootRunId;
        }
    }

    function getAssistantStreamRootRunId(container) {
        return String(
            container?.dataset?.assistantRootRunId ||
            container?.dataset?.assistantRunId ||
            ""
        ).trim();
    }

    function findAssistantStreamForRunId(runId) {
        const rootRunId = resolveRootRunId(runId);
        if (!rootRunId) {
            return null;
        }
        return conversationAssistantTurns.find(
            (container) => getAssistantStreamRootRunId(container) === rootRunId
        ) || null;
    }

    function insertBeforeDots(container, node) {
        const dots = container.querySelector(".dots");
        if (dots) container.insertBefore(node, dots);
        else container.appendChild(node);
    }

    function appendEventError(container, message) {
        const el = document.createElement("div");
        el.className = "event-error p-3 rounded-lg bg-error-container/20 border border-error/30 text-error text-sm ml-4 mb-2";
        el.textContent = `Error: ${message}`;
        container.appendChild(el);
    }

    function appendEventNotice(container, message) {
        const el = document.createElement("div");
        el.className = "event-notice p-3 rounded-lg bg-surface-container-high border border-outline-variant/20 text-on-surface-variant text-sm ml-4 mb-2";
        el.textContent = message;
        insertBeforeDots(container, el);
    }

    function getOrCreateLiveThinking(container) {
        let el = container.querySelector('div.thinking[data-live="true"]');
        if (!el) {
            el = document.createElement("div");
            el.className = "thinking px-6 py-2 border-l-2 border-primary/30 ml-4 mb-2";
            el.dataset.live = "true";
            el.innerHTML = '<div class="text-sm text-on-surface-variant italic"><span class="font-bold mr-2 text-primary">Reasoning: </span><span class="thinking-content"></span></div>';
            insertBeforeDots(container, el);
        }
        return el;
    }

    function getOrCreateLiveContent(container) {
        let el = container.querySelector('div.content[data-live="true"]');
        if (!el) {
            el = document.createElement("div");
            el.className = "content px-6 py-1 border-l-2 border-secondary/30 ml-4 mb-2";
            el.dataset.live = "true";
            el.dataset.raw = "";
            insertBeforeDots(container, el);
        }
        return el;
    }

    function appendThinkingDelta(container, delta) {
        if (!delta) return;
        const el = getOrCreateLiveThinking(container);
        const contentEl = el.querySelector(".thinking-content");
        const nextRaw = `${contentEl.textContent || ""}${delta}`;
        contentEl.textContent = nextRaw;
    }

    function appendContentDelta(container, delta) {
        if (!delta) return;
        const el = getOrCreateLiveContent(container);
        const nextRaw = `${el.dataset.raw || ""}${delta}`;
        el.dataset.raw = nextRaw;
        el.classList.remove("markdown");
        el.innerHTML = renderPlainText(nextRaw);
    }

    function parseSsePayloads(rawBuffer) {
        const normalized = rawBuffer.replace(/\r\n/g, "\n");
        const blocks = normalized.split("\n\n");
        const rest = blocks.pop() || "";
        const payloads = [];

        for (const block of blocks) {
            if (!block.trim()) continue;
            const dataLines = [];
            for (const line of block.split("\n")) {
                if (!line) continue;
                if (line.startsWith(":")) continue;
                if (line.startsWith("data:")) {
                    dataLines.push(line.slice(5).trimStart());
                }
            }
            if (!dataLines.length) continue;
            payloads.push(dataLines.join("\n").trim());
        }

        return { payloads, rest };
    }

    function renderEvent(container, event) {
        switch (event.type) {
            case "step": {
                const el = document.createElement("div");
                el.className = "step";
                el.textContent = `Step ${event.data.current} / ${event.data.max}`;
                insertBeforeDots(container, el);
                break;
            }
            case "thinking": {
                const liveEl = container.querySelector('div.thinking[data-live="true"]');
                if (liveEl) {
                    liveEl.dataset.live = "false";
                    const contentEl = liveEl.querySelector(".thinking-content");
                    contentEl.textContent = event.data.content || contentEl.textContent || "";
                } else {
                    const el = document.createElement("div");
                    el.className = "thinking px-6 py-2 border-l-2 border-primary/30 ml-4 mb-2";
                    el.innerHTML = `<div class="text-sm text-on-surface-variant italic"><span class="font-bold mr-2 text-primary">Reasoning: </span><span class="thinking-content">${escapeHtml(event.data.content || "")}</span></div>`;
                    insertBeforeDots(container, el);
                }
                break;
            }
            case "content": {
                const liveEl = container.querySelector('div.content[data-live="true"]');
                if (liveEl) {
                    liveEl.dataset.live = "false";
                    const finalText = event.data.content || liveEl.dataset.raw || "";
                    liveEl.dataset.raw = finalText;
                    liveEl.classList.add("markdown");
                    liveEl.innerHTML = renderMarkdownSafe(finalText);
                } else {
                    const el = document.createElement("div");
                    el.className = "content markdown px-6 py-1 border-l-2 border-secondary/30 ml-4 mb-2";
                    el.innerHTML = renderMarkdownSafe(event.data.content || "");
                    insertBeforeDots(container, el);
                }
                break;
            }
            case "thinking_delta": {
                appendThinkingDelta(container, event.data?.delta || "");
                break;
            }
            case "content_delta": {
                appendContentDelta(container, event.data?.delta || "");
                break;
            }
            case "tool_call": {
                const el = document.createElement("details");
                el.className = "tool w-full max-w-2xl bg-surface-container-highest/40 border border-outline-variant/20 rounded-xl overflow-hidden ml-4 mb-2 opacity-80 group";
                el.id = `tool-${event.data.id}`;
                el.innerHTML = `
                    <summary class="px-3 py-2 bg-surface-container-highest flex items-center justify-between cursor-pointer list-none select-none [&::-webkit-details-marker]:hidden">
                        <div class="flex items-center gap-2 text-on-surface-variant">
                            <span class="text-[13px]">已调用 ${escapeHtml(event.data.name || "tool")} 工具</span>
                        </div>
                    </summary>
                    <div class="p-4 font-mono text-[11px] text-primary/80 bg-black/40 border-t border-outline-variant/10">
                        <div class="flex gap-4">
                            <span class="text-on-surface-variant">ARGS:</span>
                            <span class="tool-args whitespace-pre-wrap break-words">${escapeHtml(formatToolData(event.data.arguments))}</span>
                        </div>
                    </div>
                `;
                insertBeforeDots(container, el);
                break;
            }
            case "tool_result": {
                const result = document.createElement("div");
                result.className = `mt-2 flex gap-4 tool-result`;
                const text = event.data.success ? (event.data.content || "(no output)") : (event.data.error || "Unknown error");
                const color = event.data.success ? "text-emerald-500" : "text-error";
                result.innerHTML = `<span class="${color}">${event.data.success ? "OUTPUT:" : "ERROR:"}</span><span class="${color} whitespace-pre-wrap break-words">${escapeHtml(truncate(text))}</span>`;

                const tool = container.querySelector(`#tool-${event.data.tool_call_id}`);
                if (tool) {
                    const body = tool.querySelector(".bg-black\\/40");
                    if (body) body.appendChild(result);
                    else tool.appendChild(result);
                    tool.classList.remove("opacity-80");
                } else insertBeforeDots(container, result);
                if (Array.isArray(event.data?.artifacts) && event.data.artifacts.length) {
                    const boundRunId = getAssistantStreamRootRunId(container);
                    if (boundRunId) {
                        mergeConversationArtifactsForRun(boundRunId, event.data.artifacts);
                    }
                    upsertAssistantArtifactShelf(container, event.data.artifacts);
                }
                break;
            }
            case "error":
                appendEventError(container, event.data.message || "Unknown error");
                break;
            case "interrupted":
                appendEventNotice(container, event.data?.message || "Agent run interrupted.");
                break;
            case "sub_task": {
                const subEvent = event.data.event;
                const workerIndex = event.data.worker_index ?? 0;
                const toolCallId = event.data.tool_call_id;
                let targetToolEl = null;

                if (toolCallId) {
                    targetToolEl = container.querySelector(`#tool-${toolCallId}`);
                }
                if (!targetToolEl) {
                    const toolContainers = container.querySelectorAll(".tool");
                    targetToolEl = toolContainers[toolContainers.length - 1] || null;
                }

                if (targetToolEl) {
                    let grid = targetToolEl.querySelector(".sub-agent-grid");
                    if (!grid) {
                        grid = document.createElement("div");
                        grid.className = "sub-agent-grid flex flex-row overflow-x-auto gap-4 mt-4 w-full px-4 pb-6 items-start";
                        targetToolEl.appendChild(grid);
                    }

                    let panel = grid.querySelector(`[data-worker-index="${workerIndex}"]`);
                    if (!panel) {
                        panel = document.createElement("div");
                        panel.className = "sub-agent-panel bg-surface-container-low/80 border border-outline-variant/30 rounded-xl p-4 flex flex-col gap-2 w-[300px] shrink-0 overflow-hidden";
                        panel.dataset.workerIndex = workerIndex;
                        panel.innerHTML = `
                            <div class="sub-agent-header flex justify-between items-center border-b border-outline-variant/10 pb-2 mb-2">
                                <span class="text-[10px] font-mono text-cyan-400 font-bold uppercase tracking-wider">Sub-Agent #${workerIndex + 1}</span>
                                <span class="status text-[10px] font-mono text-on-surface-variant uppercase bg-surface-container-highest px-2 py-0.5 rounded">Running</span>
                            </div>
                            <div class="stream w-full"></div>
                        `;
                        grid.appendChild(panel);
                    }

                    const subStream = panel.querySelector(".stream");
                    const dots = container.querySelector(".dots");
                    if (dots) dots.remove();

                    if (subEvent.type === "done" || subEvent.type === "error") {
                        panel.querySelector(".status").textContent = subEvent.type === "done" ? "Completed" : "Error";
                    }

                    renderEvent(subStream, subEvent);
                    if (dots) container.appendChild(dots);
                }
                break;
            }
            default:
                break;
        }
        if (activeRunId) {
            scheduleRunRefresh();
        }
        scrollToBottom();
    }

    function renderPersistedMessages(messages) {
        $messages.innerHTML = "";
        resetConversationAssistantTurns();
        let currentAssistantContainer = null;

        messages.forEach((message) => {
            if (message.role === "system") {
                return;
            }

            if (message.role === "user") {
                appendMessage("user", message.content);
                currentAssistantContainer = null;
                return;
            }

            if (message.role === "assistant") {
                if (!currentAssistantContainer) {
                    currentAssistantContainer = appendAssistantContainer(false);
                }
                if (message.run_id && isFeatureEnabled("enable_durable_runs")) {
                    setAssistantStreamRunId(currentAssistantContainer, message.run_id);
                }
                if (message.thinking) {
                    renderEvent(currentAssistantContainer, {
                        type: "thinking",
                        data: { content: message.thinking },
                    });
                }
                if (message.content) {
                    renderEvent(currentAssistantContainer, {
                        type: "content",
                        data: { content: messageText(message.content) },
                    });
                }
                (message.tool_calls || []).forEach((toolCall) => {
                    renderEvent(currentAssistantContainer, {
                        type: "tool_call",
                        data: {
                            id: toolCall.id,
                            name: toolCall.function?.name,
                            arguments: toolCall.function?.arguments || {},
                        },
                    });
                });
                return;
            }

            if (message.role === "tool") {
                if (!currentAssistantContainer) {
                    currentAssistantContainer = appendAssistantContainer(false);
                }
                if (message.run_id && isFeatureEnabled("enable_durable_runs")) {
                    setAssistantStreamRunId(currentAssistantContainer, message.run_id);
                }
                const text = messageText(message.content);
                const isError = text.startsWith("Error:");
                renderEvent(currentAssistantContainer, {
                    type: "tool_result",
                    data: {
                        tool_call_id: message.tool_call_id,
                        name: message.name,
                        success: !isError,
                        content: isError ? null : text,
                        error: isError ? text.replace(/^Error:\s*/, "") : null,
                        artifacts: message.artifacts || undefined,
                    },
                });
            }
        });

        showChatView();
    }

    async function switchSession(nextSessionId) {
        if (!nextSessionId || isStreaming) {
            return;
        }

        setStatus("正在加载会话");
        const detail = await fetchJson(`/api/sessions/${nextSessionId}/messages`);
        sessionId = nextSessionId;
        setActiveAgent(detail.agent_id);
        setSessionGroupCollapsed(detail.agent_id, false);
        renderSessionList();
        clearComposerAttachments();
        renderPersistedMessages(detail.messages || []);
        if (isFeatureEnabled("enable_durable_runs")) {
            await refreshRunSidebar({ preserveSelection: false });
            syncLatestAssistantArtifactShelfFromSelectedRun();
        } else {
            clearRunState();
        }
        await refreshSharedContextPanel(sessionId);
        await refreshSessionUploads(sessionId);
        await refreshMemoryViewData();
        setStatus("会话已就绪");
        setComposerState("Ready");
    }

    async function deleteSession(targetSessionId) {
        if (!targetSessionId || isStreaming) {
            return;
        }

        const confirmed = window.confirm("删除后将无法恢复这条会话，确认继续吗？");
        if (!confirmed) {
            return;
        }

        const response = await fetch(`/api/sessions/${targetSessionId}`, { method: "DELETE" });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        if (sessionId === targetSessionId) {
            sessionId = null;
            $messages.innerHTML = "";
            resetConversationAssistantTurns();
            clearComposerAttachments();
            clearRunState();
            clearSharedContextPanel();
            clearSessionUploadsPanel();
        }
        await refreshSessionList();
        if (sessions.length) {
            await switchSession(sessions[0].session_id);
        } else {
            await createSession({ openChat: true });
        }
    }

    async function requestInterrupt() {
        if (!sessionId || !isStreaming || interruptPending) {
            return;
        }

        interruptPending = true;
        setStatus("Interrupt requested");
        setComposerState("Interrupting");
        updateComposerActionButton();

        try {
            await fetchJson(`/api/sessions/${sessionId}/interrupt`, {
                method: "POST",
            });
            scheduleRunRefresh();
        } catch (error) {
            console.error("Failed to interrupt session:", error);
            interruptPending = false;
            setStatus("Interrupt failed");
            setComposerState("Streaming");
            updateComposerActionButton();
        }
    }

    async function sendMessage(textOverride = null) {
        const text = (textOverride ?? $input.value).trim();
        const readyAttachments = getReadyComposerAttachments();
        if (isStreaming) return;
        if (hasBlockingComposerAttachments()) {
            setStatus("请先等待上传完成，或移除失败的附件");
            updateComposerActionButton();
            return;
        }
        if (!text && !readyAttachments.length) return;

        if (!sessionId) {
            await createSession({ openChat: true });
        }
        if (!sessionId) return;

        showChatView();

        const userMessageContent = buildOutgoingUserContent(text, readyAttachments);
        appendMessage("user", userMessageContent);
        $input.value = "";
        clearComposerAttachments();
        autoResizeInput();
        updateComposerActionButton();
        setStreaming(true);
        const assistantContainer = appendAssistantContainer(true);
        activeAssistantContainer = assistantContainer;
        let currentRunId = null;

        try {
            const res = await fetch(`/api/sessions/${sessionId}/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: text,
                    attachment_ids: readyAttachments
                        .map((attachment) => attachment.upload?.id || "")
                        .filter(Boolean),
                }),
            });
            if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
            currentRunId = res.headers.get("X-Run-ID");
            if (currentRunId && isFeatureEnabled("enable_durable_runs")) {
                setAssistantStreamRunId(assistantContainer, currentRunId);
                activeRunId = currentRunId;
                selectedRunId = currentRunId;
                setSidebarTab("runs");
                await refreshRunSidebar({ focusRunId: currentRunId, preserveSelection: false });
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let streamCompleted = false;

            while (!streamCompleted) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const parsed = parseSsePayloads(buffer);
                buffer = parsed.rest;

                for (const payload of parsed.payloads) {
                    if (!payload) continue;
                    if (payload === "[DONE]") {
                        streamCompleted = true;
                        break;
                    }
                    try {
                        const event = JSON.parse(payload);
                        renderEvent(assistantContainer, event);
                        if (currentRunId) {
                            setAssistantStreamRunId(assistantContainer, currentRunId);
                            activeRunId = currentRunId;
                        }
                        if (hasShareContextUpdate(event)) {
                            scheduleSharedContextRefresh();
                        }
                    } catch (error) {
                        console.error("Failed to parse event:", error);
                    }
                }
            }

            if (!streamCompleted && buffer.trim()) {
                const parsed = parseSsePayloads(`${buffer}\n\n`);
                for (const payload of parsed.payloads) {
                    if (!payload || payload === "[DONE]") continue;
                    try {
                        const event = JSON.parse(payload);
                        renderEvent(assistantContainer, event);
                        if (currentRunId) {
                            setAssistantStreamRunId(assistantContainer, currentRunId);
                            activeRunId = currentRunId;
                        }
                        if (hasShareContextUpdate(event)) {
                            scheduleSharedContextRefresh();
                        }
                    } catch (error) {
                        console.error("Failed to parse trailing event:", error);
                    }
                }
            }
        } catch (error) {
            appendEventError(assistantContainer, `连接失败：${error.message}`);
        }

        const dots = assistantContainer.querySelector(".dots");
        if (dots) dots.remove();
        setStreaming(false);
        if (isFeatureEnabled("enable_durable_runs")) {
            await refreshRunSidebar({ focusRunId: currentRunId || activeRunId });
            syncLatestAssistantArtifactShelfFromSelectedRun(currentRunId || activeRunId);
        }
        await refreshSessionList(sessionId);
        await refreshSharedContextPanel(sessionId);
        await refreshSessionUploads(sessionId);
        await refreshMemoryViewData();
        scrollToBottom();
    }

    function startSendMessage(textOverride = null) {
        const runPromise = sendMessage(textOverride);
        activeStreamPromise = runPromise;
        runPromise.finally(() => {
            if (activeStreamPromise === runPromise) {
                activeStreamPromise = null;
            }
        });
        return runPromise;
    }

    async function requestHumanIntervention() {
        const interventionText = $input.value.trim();
        if (!interventionText) {
            await requestInterrupt();
            return;
        }
        if (!sessionId || !isStreaming || interruptPending) {
            return;
        }

        const targetSessionId = sessionId;
        interruptPending = true;
        setStatus("Human intervention requested");
        setComposerState("Human in loop");
        updateComposerActionButton();
        if (activeAssistantContainer) {
            appendEventNotice(
                activeAssistantContainer,
                `Human intervention requested. Redirecting with: ${truncate(interventionText, 180)}`
            );
        }

        try {
            await fetchJson(`/api/sessions/${sessionId}/interrupt`, {
                method: "POST",
            });
        } catch (error) {
            console.error("Failed to request human intervention:", error);
            interruptPending = false;
            setStatus("Intervention failed");
            setComposerState("Streaming");
            updateComposerActionButton();
            return;
        }

        const runningTurn = activeStreamPromise;
        if (runningTurn) {
            try {
                await runningTurn;
            } catch (error) {
                console.error("Interrupted run failed while waiting for human intervention:", error);
            }
        }

        if (sessionId !== targetSessionId) {
            return;
        }

        setStatus("Applying human guidance");
        setComposerState("Resuming");
        await startSendMessage(interventionText);
    }

    function scrollToBottom(force = false) {
        if (!force && userScrolledUp) return;
        requestAnimationFrame(() => {
            $messages.scrollTop = $messages.scrollHeight;
        });
    }

    async function bootstrap() {
        try {
            // 先校验登录态；已登录后再检查当前账号是否绑定了可用 API Key
            try {
                const authResp = await fetch("/api/auth/me");
                if (authResp.status === 401) {
                    const currentPath = encodeURIComponent(window.location.href);
                    window.location.href = `/static/login.html?redirect=${currentPath}`;
                    return;
                }
                if (authResp.ok) {
                    const authData = await authResp.json();
                    currentAccount = authData?.account || null;
                    applyAccountToUI(currentAccount);
                }

                const setupResp = await fetch("/api/setup/status");
                if (setupResp.ok) {
                    const setupData = await setupResp.json();
                    if (setupData?.setup_required) {
                        const currentPath = encodeURIComponent(window.location.href);
                        window.location.href = `/static/setup.html?redirect=${currentPath}`;
                        return;
                    }
                }
            } catch (authError) {
                // 网络异常时不阻断，继续尝试加载（服务可能关闭鉴权）
                console.warn("Auth check failed:", authError);
            }

            renderHome();
            autoResizeInput();
            updateComposerActionButton();
            setStatus("Loading sessions");
            setComposerState("Loading");
            setActiveAgent(DEFAULT_AGENT_ID);
            await refreshFeatureFlags();
            await refreshAgentCatalog();
            await refreshSessionList();
            if (sessions.length) {
                sessionId = sessions[0].session_id;
                setActiveAgent(sessions[0].agent_id);
                setSessionGroupCollapsed(sessions[0].agent_id, false);
            } else {
                sessionId = null;
            }
            clearComposerAttachments();
            clearSharedContextPanel();
            clearSessionUploadsPanel();
            if (isFeatureEnabled("enable_durable_runs")) {
                await refreshRunSidebar({ preserveSelection: false });
            } else {
                clearRunState();
            }
            if (sessionId) {
                await refreshSessionUploads(sessionId);
            }
            await refreshMemoryViewData();
            renderHome();
        } catch (error) {
            console.error("Failed to bootstrap app:", error);
            setStatus("会话加载失败");
            setComposerState("Error");
        }
    }

    // 将账号信息渲染到 Header UI
    function applyAccountToUI(account) {
        if (!account) return;

        const $avatar = document.getElementById("account-avatar");
        const $usernameLabel = document.getElementById("account-username-label");
        const $dropdownDisplayName = document.getElementById("dropdown-display-name");
        const $dropdownUsername = document.getElementById("dropdown-username");
        const $rootBadge = document.getElementById("dropdown-root-badge");
        const $btnApiConfig = document.getElementById("btn-api-config");
        const $btnAgentRouting = document.getElementById("btn-agent-routing");
        const $btnAccountAdmin = document.getElementById("btn-account-admin");

        const displayName = account.display_name || account.username || "?";
        const username = account.username || "";
        const firstChar = (displayName[0] || "?").toUpperCase();

        if ($avatar) $avatar.textContent = firstChar;
        if ($usernameLabel) $usernameLabel.textContent = username;
        if ($dropdownDisplayName) $dropdownDisplayName.textContent = displayName;
        if ($dropdownUsername) $dropdownUsername.textContent = `@${username}`;

        // API 配置对所有账号开放；账号管理仍只对 root 开放
        if ($rootBadge) $rootBadge.classList.toggle("hidden", !account.is_root);
        if ($btnApiConfig) $btnApiConfig.classList.remove("hidden");
        if ($btnAgentRouting) $btnAgentRouting.classList.remove("hidden");
        if ($btnAccountAdmin) $btnAccountAdmin.classList.toggle("hidden", !account.is_root);
    }

    // 退出登录
    async function handleLogout() {
        try {
            await fetch("/api/auth/logout", { method: "POST" });
        } catch (err) {
            console.warn("Logout request failed:", err);
        }
        window.location.href = "/static/login.html";
    }

    function openApiConfigPage() {
        const currentPath = encodeURIComponent(window.location.href);
        window.location.href = `/static/setup.html?redirect=${currentPath}`;
    }

    function openAgentRoutingPage() {
        const currentPath = encodeURIComponent(window.location.href);
        window.location.href = `/static/agent_routing.html?redirect=${currentPath}`;
    }

    // 修改密码弹层
    function showChangePasswordDialog() {
        const existing = document.getElementById("change-password-modal");
        if (existing) {
            existing.classList.remove("hidden");
            existing.classList.add("flex");
            return;
        }

        const modal = document.createElement("div");
        modal.id = "change-password-modal";
        modal.className = "fixed inset-0 z-[200] flex items-center justify-center bg-black/70 px-4 backdrop-blur-sm";
        modal.innerHTML = `
            <div class="w-full max-w-sm rounded-3xl border border-outline-variant/20 bg-surface-container shadow-[0_32px_80px_rgba(0,0,0,0.5)] overflow-hidden">
                <div class="px-6 py-5 border-b border-white/5">
                    <div class="text-[10px] font-mono uppercase tracking-[0.22em] text-primary/80 mb-1">账号设置</div>
                    <h3 class="text-base font-semibold text-on-surface">修改密码</h3>
                </div>
                <div class="px-6 py-5 space-y-4">
                    <div>
                        <label class="block text-xs font-mono uppercase tracking-widest text-on-surface-variant mb-2">当前密码</label>
                        <input id="cp-current" type="password" placeholder="当前密码"
                            class="w-full bg-surface-container-highest border border-outline-variant/30 rounded-xl px-4 py-2.5 text-sm text-on-surface placeholder-on-surface-variant/50 focus:outline-none focus:border-primary/50" />
                    </div>
                    <div>
                        <label class="block text-xs font-mono uppercase tracking-widest text-on-surface-variant mb-2">新密码</label>
                        <input id="cp-new" type="password" placeholder="至少 8 位"
                            class="w-full bg-surface-container-highest border border-outline-variant/30 rounded-xl px-4 py-2.5 text-sm text-on-surface placeholder-on-surface-variant/50 focus:outline-none focus:border-primary/50" />
                    </div>
                    <div id="cp-error" class="hidden rounded-xl border border-error/30 bg-error/10 px-3 py-2 text-xs text-error"></div>
                </div>
                <div class="px-6 pb-5 flex gap-3 justify-end">
                    <button id="cp-cancel" class="px-4 py-2 rounded-xl border border-outline-variant/20 text-sm text-on-surface-variant hover:text-on-surface transition-colors">取消</button>
                    <button id="cp-submit" class="px-4 py-2 rounded-xl bg-primary text-on-primary text-sm font-bold hover:bg-primary-container transition-colors">确认修改</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        const close = () => {
            modal.classList.add("hidden");
            modal.classList.remove("flex");
        };

        document.getElementById("cp-cancel").onclick = close;
        modal.addEventListener("click", (e) => { if (e.target === modal) close(); });
        document.getElementById("cp-submit").onclick = async () => {
            const current = document.getElementById("cp-current").value;
            const newPwd = document.getElementById("cp-new").value;
            const errEl = document.getElementById("cp-error");
            errEl.classList.add("hidden");

            if (!current || !newPwd) {
                errEl.textContent = "请填写当前密码和新密码。";
                errEl.classList.remove("hidden");
                return;
            }

            try {
                const resp = await fetch("/api/auth/change-password", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ current_password: current, new_password: newPwd }),
                });
                if (resp.ok) {
                    close();
                    setStatus("密码已更新");
                } else {
                    const data = await resp.json().catch(() => ({}));
                    errEl.textContent = data?.detail || "修改失败，请重试。";
                    errEl.classList.remove("hidden");
                }
            } catch (err) {
                errEl.textContent = "网络错误，请重试。";
                errEl.classList.remove("hidden");
            }
        };
    }

    // 账号管理弹层（root 专用）
    async function showAccountAdminDialog() {
        const existing = document.getElementById("account-admin-modal");
        if (existing) {
            existing.classList.remove("hidden");
            existing.classList.add("flex");
            await refreshAccountAdminList();
            return;
        }

        const modal = document.createElement("div");
        modal.id = "account-admin-modal";
        modal.className = "fixed inset-0 z-[200] flex items-center justify-center bg-black/70 px-4 backdrop-blur-sm";
        modal.innerHTML = `
            <div class="w-full max-w-lg rounded-3xl border border-outline-variant/20 bg-surface-container shadow-[0_32px_80px_rgba(0,0,0,0.5)] overflow-hidden flex flex-col max-h-[80vh]">
                <div class="px-6 py-5 border-b border-white/5 flex items-center justify-between shrink-0">
                    <div>
                        <div class="text-[10px] font-mono uppercase tracking-[0.22em] text-amber-200/80 mb-1">Root 管理后台</div>
                        <h3 class="text-base font-semibold text-on-surface">账号管理</h3>
                    </div>
                    <button id="aa-close" class="w-8 h-8 flex items-center justify-center rounded-full border border-outline-variant/20 text-on-surface-variant hover:text-on-surface transition-colors">
                        <span class="material-symbols-outlined text-sm">close</span>
                    </button>
                </div>

                <div class="flex-1 overflow-y-auto p-4 space-y-3" id="aa-account-list">
                    <div class="text-sm text-on-surface-variant text-center py-8">加载中…</div>
                </div>

                <div class="px-6 pb-5 pt-4 border-t border-white/5 shrink-0">
                    <div class="text-xs font-mono uppercase tracking-widest text-on-surface-variant mb-3">新建账号</div>
                    <div class="grid grid-cols-2 gap-3">
                        <input id="aa-new-username" type="text" placeholder="用户名"
                            class="bg-surface-container-highest border border-outline-variant/30 rounded-xl px-3 py-2 text-sm text-on-surface placeholder-on-surface-variant/50 focus:outline-none focus:border-primary/50" />
                        <input id="aa-new-password" type="password" placeholder="密码（至少8位）"
                            class="bg-surface-container-highest border border-outline-variant/30 rounded-xl px-3 py-2 text-sm text-on-surface placeholder-on-surface-variant/50 focus:outline-none focus:border-primary/50" />
                    </div>
                    <div id="aa-error" class="hidden mt-2 rounded-xl border border-error/30 bg-error/10 px-3 py-2 text-xs text-error"></div>
                    <button id="aa-create-btn" class="mt-3 w-full px-4 py-2 rounded-xl bg-primary text-on-primary text-sm font-bold hover:bg-primary-container transition-colors flex items-center justify-center gap-2">
                        <span class="material-symbols-outlined text-sm">person_add</span>
                        <span>创建账号</span>
                    </button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        const close = () => {
            modal.classList.add("hidden");
            modal.classList.remove("flex");
        };
        document.getElementById("aa-close").onclick = close;
        modal.addEventListener("click", (e) => { if (e.target === modal) close(); });

        document.getElementById("aa-create-btn").onclick = async () => {
            const username = document.getElementById("aa-new-username").value.trim();
            const password = document.getElementById("aa-new-password").value;
            const errEl = document.getElementById("aa-error");
            errEl.classList.add("hidden");

            if (!username || !password) {
                errEl.textContent = "请填写用户名和密码。";
                errEl.classList.remove("hidden");
                return;
            }
            try {
                const resp = await fetch("/api/accounts", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ username, password }),
                });
                if (resp.ok) {
                    document.getElementById("aa-new-username").value = "";
                    document.getElementById("aa-new-password").value = "";
                    await refreshAccountAdminList();
                } else {
                    const data = await resp.json().catch(() => ({}));
                    errEl.textContent = data?.detail || "创建失败，请重试。";
                    errEl.classList.remove("hidden");
                }
            } catch (err) {
                errEl.textContent = "网络错误，请重试。";
                errEl.classList.remove("hidden");
            }
        };

        await refreshAccountAdminList();
    }

    // 刷新账号管理列表
    async function refreshAccountAdminList() {
        const listEl = document.getElementById("aa-account-list");
        if (!listEl) return;
        try {
            const accounts = await fetchJson("/api/accounts");
            if (!Array.isArray(accounts) || !accounts.length) {
                listEl.innerHTML = `<div class="text-sm text-on-surface-variant text-center py-8">暂无账号。</div>`;
                return;
            }
            listEl.innerHTML = accounts.map((acc) => `
                <article class="rounded-2xl border border-outline-variant/15 bg-surface-container-high/60 px-4 py-3 flex items-center gap-3">
                    <div class="w-9 h-9 rounded-full border border-primary/20 bg-primary/10 flex items-center justify-center text-primary font-bold text-sm uppercase shrink-0">
                        ${escapeHtml((acc.display_name || acc.username || "?")[0].toUpperCase())}
                    </div>
                    <div class="flex-1 min-w-0">
                        <div class="flex items-center gap-2">
                            <span class="text-sm font-semibold text-on-surface truncate">${escapeHtml(acc.display_name || acc.username)}</span>
                            ${acc.is_root ? `<span class="text-[9px] font-mono uppercase border border-amber-400/30 bg-amber-400/10 text-amber-200 px-1.5 py-0.5 rounded-full">ROOT</span>` : ""}
                            ${acc.status === "disabled" ? `<span class="text-[9px] font-mono uppercase border border-error/30 bg-error/10 text-error px-1.5 py-0.5 rounded-full">已禁用</span>` : ""}
                        </div>
                        <div class="text-xs text-on-surface-variant font-mono">@${escapeHtml(acc.username)}</div>
                    </div>
                    <div class="flex items-center gap-1 shrink-0">
                        ${!acc.is_root ? `
                        <button
                            class="px-2 py-1 rounded-lg border border-outline-variant/20 text-[10px] font-mono uppercase text-on-surface-variant hover:text-on-surface hover:bg-surface-container-highest transition-colors aa-toggle-btn"
                            data-account-id="${escapeHtml(acc.id)}"
                            data-action="toggle"
                            data-current-status="${escapeHtml(acc.status)}"
                        >${acc.status === "active" ? "禁用" : "启用"}</button>
                        <button
                            class="px-2 py-1 rounded-lg border border-outline-variant/20 text-[10px] font-mono uppercase text-on-surface-variant hover:text-primary hover:border-primary/30 transition-colors aa-reset-btn"
                            data-account-id="${escapeHtml(acc.id)}"
                            data-action="reset-password"
                        >重置密码</button>` : ""}
                    </div>
                </article>
            `).join("");

            // 绑定按钮事件
            listEl.querySelectorAll(".aa-toggle-btn").forEach((btn) => {
                btn.onclick = async () => {
                    const accountId = btn.dataset.accountId;
                    const currentStatus = btn.dataset.currentStatus;
                    const newStatus = currentStatus === "active" ? "disabled" : "active";
                    try {
                        const resp = await fetch(`/api/accounts/${encodeURIComponent(accountId)}`, {
                            method: "PATCH",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ status: newStatus }),
                        });
                        if (resp.ok) {
                            await refreshAccountAdminList();
                        } else {
                            const data = await resp.json().catch(() => ({}));
                            alert(data?.detail || "操作失败。");
                        }
                    } catch {
                        alert("网络错误，请重试。");
                    }
                };
            });

            listEl.querySelectorAll(".aa-reset-btn").forEach((btn) => {
                btn.onclick = async () => {
                    const accountId = btn.dataset.accountId;
                    const newPwd = prompt("请输入新密码（至少 8 位）：");
                    if (!newPwd || newPwd.length < 8) {
                        if (newPwd !== null) alert("密码长度不符合要求（至少 8 位）。");
                        return;
                    }
                    try {
                        const resp = await fetch(`/api/accounts/${encodeURIComponent(accountId)}/reset-password`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ new_password: newPwd }),
                        });
                        if (resp.ok) {
                            alert("密码已重置。");
                        } else {
                            const data = await resp.json().catch(() => ({}));
                            alert(data?.detail || "操作失败。");
                        }
                    } catch {
                        alert("网络错误，请重试。");
                    }
                };
            });
        } catch (err) {
            listEl.innerHTML = `<div class="text-sm text-error text-center py-8">加载失败：${escapeHtml(err.message || "未知错误")}</div>`;
        }
    }

    // 账号下拉菜单交互
    const $accountMenuBtn = document.getElementById("account-menu-btn");
    const $accountDropdown = document.getElementById("account-dropdown");

    if ($accountMenuBtn && $accountDropdown) {
        // 点击头像按钮切换下拉菜单
        $accountMenuBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            const isOpen = !$accountDropdown.classList.contains("hidden");
            $accountDropdown.classList.toggle("hidden", isOpen);
        });

        // 点击页面其他区域关闭菜单
        document.addEventListener("click", (e) => {
            if (!$accountMenuBtn.contains(e.target) && !$accountDropdown.contains(e.target)) {
                $accountDropdown.classList.add("hidden");
            }
        });
    }

    // 退出登录
    const $btnLogout = document.getElementById("btn-logout");
    if ($btnLogout) {
        $btnLogout.addEventListener("click", () => {
            if ($accountDropdown) $accountDropdown.classList.add("hidden");
            handleLogout();
        });
    }

    // 修改密码
    const $btnChangePassword = document.getElementById("btn-change-password");
    if ($btnChangePassword) {
        $btnChangePassword.addEventListener("click", () => {
            if ($accountDropdown) $accountDropdown.classList.add("hidden");
            showChangePasswordDialog();
        });
    }

    // 账号管理（root 专用）
    const $btnApiConfig = document.getElementById("btn-api-config");
    if ($btnApiConfig) {
        $btnApiConfig.addEventListener("click", () => {
            if ($accountDropdown) $accountDropdown.classList.add("hidden");
            openApiConfigPage();
        });
    }

    const $btnAgentRouting = document.getElementById("btn-agent-routing");
    if ($btnAgentRouting) {
        $btnAgentRouting.addEventListener("click", () => {
            if ($accountDropdown) $accountDropdown.classList.add("hidden");
            openAgentRoutingPage();
        });
    }

    const $btnAccountAdmin = document.getElementById("btn-account-admin");
    if ($btnAccountAdmin) {
        $btnAccountAdmin.addEventListener("click", () => {
            if ($accountDropdown) $accountDropdown.classList.add("hidden");
            showAccountAdminDialog();
        });
    }

    $messages.addEventListener("scroll", () => {
        const atBottom = $messages.scrollHeight - $messages.scrollTop - $messages.clientHeight < 100;
        userScrolledUp = !atBottom;
    });
    $messages.addEventListener("click", (event) => {
        const previewButton = event.target.closest("[data-file-preview-target]");
        if (!previewButton) {
            return;
        }
        openFilePreviewFromDataset(previewButton.dataset);
    });

    if ($composerAttachments) {
        $composerAttachments.addEventListener("click", (event) => {
            const removeButton = event.target.closest("[data-remove-composer-attachment]");
            if (!removeButton) {
                return;
            }
            removeComposerAttachment(removeButton.dataset.removeComposerAttachment);
        });
    }

    $input.addEventListener("input", () => {
        autoResizeInput();
        updateComposerActionButton();
    });
    $input.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            if (isStreaming) {
                if ($input.value.trim()) {
                    requestHumanIntervention();
                } else {
                    requestInterrupt();
                }
                return;
            }
            startSendMessage();
        }
    });
    if ($uploadTriggerBtn) {
        $uploadTriggerBtn.addEventListener("click", () => {
            if ($uploadInput && !$uploadTriggerBtn.disabled) {
                $uploadInput.click();
            }
        });
    }
    if ($uploadInput) {
        $uploadInput.addEventListener("change", async (event) => {
            const files = Array.from(event.target.files || []);
            event.target.value = "";
            await handleSelectedFiles(files);
        });
    }
    if ($sessionUploadsList) {
        $sessionUploadsList.addEventListener("click", (event) => {
            const previewButton = event.target.closest("[data-file-preview-target]");
            if (previewButton) {
                openFilePreviewFromDataset(previewButton.dataset);
                return;
            }
            const attachButton = event.target.closest("[data-upload-action='attach']");
            if (!attachButton) {
                return;
            }
            const upload = sessionUploads.find((item) => item.id === attachButton.dataset.uploadId);
            if (!upload) {
                return;
            }
            addUploadToComposer(upload, "session");
            setStatus(`已附加文件：${upload.original_name || upload.safe_name || upload.id}`);
        });
    }
    if ($sessionUploadsRefreshBtn) {
        $sessionUploadsRefreshBtn.addEventListener("click", () => {
            refreshSessionUploads(sessionId).catch((error) => {
                console.error("Failed to refresh session uploads:", error);
            });
        });
    }
    if ($chatView) {
        $chatView.addEventListener("dragenter", (event) => {
            if (!event.dataTransfer?.types?.includes("Files")) {
                return;
            }
            event.preventDefault();
            uploadDragDepth += 1;
            if (event.dataTransfer) {
                event.dataTransfer.dropEffect = "copy";
            }
            setUploadDropzoneActive(true);
        });
        $chatView.addEventListener("dragover", (event) => {
            if (!event.dataTransfer?.types?.includes("Files")) {
                return;
            }
            event.preventDefault();
            if (event.dataTransfer) {
                event.dataTransfer.dropEffect = "copy";
            }
            setUploadDropzoneActive(true);
        });
        $chatView.addEventListener("dragleave", (event) => {
            if (!event.dataTransfer?.types?.includes("Files")) {
                return;
            }
            uploadDragDepth = Math.max(0, uploadDragDepth - 1);
            if (uploadDragDepth === 0) {
                setUploadDropzoneActive(false);
            }
        });
        $chatView.addEventListener("drop", (event) => {
            if (!event.dataTransfer?.files?.length) {
                return;
            }
            event.preventDefault();
            uploadDragDepth = 0;
            setUploadDropzoneActive(false);
            handleSelectedFiles(event.dataTransfer.files).catch((error) => {
                console.error("Failed to handle dropped files:", error);
            });
        });
    }
    document.addEventListener("dragend", () => {
        uploadDragDepth = 0;
        setUploadDropzoneActive(false);
    });
    document.addEventListener("drop", () => {
        uploadDragDepth = 0;
        setUploadDropzoneActive(false);
    });
    $sendBtn.addEventListener("click", () => {
        if (isStreaming) {
            requestInterrupt();
            return;
        }
        startSendMessage();
    });
    if ($interveneBtn) {
        $interveneBtn.addEventListener("click", () => {
            requestHumanIntervention();
        });
    }
    $sidebarTabButtons.forEach((button) => {
        button.addEventListener("click", () => {
            setSidebarTab(button.dataset.sidebarTab);
        });
    });
    if ($runPanelRefreshBtn) {
        $runPanelRefreshBtn.addEventListener("click", () => {
            handleRunAction("refresh").catch((error) => {
                console.error("Failed to refresh run panel:", error);
            });
        });
    }
    if ($filePreviewCloseBtn) {
        $filePreviewCloseBtn.addEventListener("click", () => {
            closeFilePreview();
        });
    }
    if ($filePreviewModal) {
        $filePreviewModal.addEventListener("click", (event) => {
            if (event.target === $filePreviewModal) {
                closeFilePreview();
            }
        });
    }
    if ($runsPanel) {
        $runsPanel.addEventListener("click", handleRunSurfaceClick);
    }
    if ($memoryView) {
        $memoryView.addEventListener("click", (event) => {
            handleMemoryViewClick(event).catch((error) => {
                console.error("Failed to handle memory view action:", error);
            });
        });
        $memoryView.addEventListener("submit", (event) => {
            handleMemoryViewSubmit(event).catch((error) => {
                console.error("Failed to handle memory view submit:", error);
            });
        });
    }
    if ($approvalInboxList) {
        $approvalInboxList.addEventListener("click", handleRunSurfaceClick);
    }
    if ($headerOpenRunsBtn) {
        $headerOpenRunsBtn.addEventListener("click", () => {
            setSidebarTab("runs");
            togglePanel("shared-context-sidebar", false);
        });
    }
    if ($homeBtn) $homeBtn.addEventListener("click", renderHome);
    if ($newSessionBtn) $newSessionBtn.addEventListener("click", () => createSessionForSelectedAgent({ openChat: true }));
    $sidebarNewSessionBtn.addEventListener("click", () => createSessionForSelectedAgent({ openChat: true }));

    if ($logoHomeBtn) {
        $logoHomeBtn.addEventListener("click", renderHome);
    }
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && $filePreviewModal && !$filePreviewModal.classList.contains("hidden")) {
            closeFilePreview();
            return;
        }
    });
    document.addEventListener("keydown", (event) => {
        if (event.key !== "Enter") return;
        const searchInput = document.getElementById("skillSearchKeywordInput");
        if (!searchInput || event.target !== searchInput) return;
        event.preventDefault();
        window.searchAgentSkills();
    });

    function updateExpandButtons() {
        if (panels["pane-session"] && panels["pane-session"].btn) {
            let leftBase = 0;
            if (panels["pane-nav"] && panels["pane-nav"].el.style.display !== "none") {
                leftBase += panels["pane-nav"].el.getBoundingClientRect().width;
            }
            panels["pane-session"].btn.style.left = leftBase + "px";
        }
    }

    function handleRunSurfaceClick(event) {
        const previewButton = event.target.closest("[data-file-preview-target]");
        if (previewButton) {
            openFilePreviewFromDataset(previewButton.dataset);
            return;
        }

        const actionButton = event.target.closest("[data-run-action]");
        if (actionButton) {
            handleRunAction(actionButton.dataset.runAction, actionButton.dataset.runId).catch((error) => {
                console.error("Failed to handle run action:", error);
            });
            return;
        }

        const approvalButton = event.target.closest("[data-approval-action]");
        if (!approvalButton) return;
        handleApprovalAction(
            approvalButton.dataset.approvalAction,
            approvalButton.dataset.approvalId,
            approvalButton.dataset.runId
        ).catch((error) => {
            console.error("Failed to handle approval action:", error);
        });
    }

    // --- Layout Manage: Resizing and Toggling ---
    const $paneNav = document.getElementById("pane-nav");
    const $paneSession = document.getElementById("pane-session");
    const $paneContext = document.getElementById("shared-context-sidebar");

    const $btnExpandNav = document.getElementById("btn-expand-nav");
    const $btnExpandSession = document.getElementById("btn-expand-session");
    const $btnExpandContext = document.getElementById("btn-expand-context");

    const panels = {
        "pane-nav": { el: $paneNav, btn: $btnExpandNav, min: 200, max: 400, default: 256, isLeft: true },
        "pane-session": { el: $paneSession, btn: $btnExpandSession, min: 250, max: 500, default: 320, isLeft: true },
        "shared-context-sidebar": { el: $paneContext, btn: $btnExpandContext, min: 280, max: 560, default: 360, isLeft: false }
    };

    function togglePanel(id, forceState) {
        const p = panels[id];
        if (!p || !p.el) return;
        const isCurrentlyCollapsed = p.el.style.display === "none";
        const willCollapse = forceState !== undefined ? forceState : !isCurrentlyCollapsed;

        if (willCollapse) {
            // Collapse
            p.el.style.display = "none";
            if (p.btn) p.btn.classList.remove("hidden");
            localStorage.setItem(`pane_collapsed_${id}`, "true");
        } else {
            // Uncollapse
            p.el.style.display = "";
            let savedW = localStorage.getItem(`pane_width_${id}`);
            p.el.style.width = (savedW && savedW > p.min ? savedW : p.default) + "px";
            if (p.btn) p.btn.classList.add("hidden");
            localStorage.setItem(`pane_collapsed_${id}`, "false");
        }
        updateExpandButtons();
    }

    if (localStorage.getItem("pane_collapsed_shared-context-sidebar") === null) {
        localStorage.setItem("pane_collapsed_shared-context-sidebar", "true");
    }

    // Initialize widths and states
    Object.keys(panels).forEach(id => {
        const p = panels[id];
        if (!p || !p.el) return;
        const isCollapsed = localStorage.getItem(`pane_collapsed_${id}`) === "true";
        if (isCollapsed) {
            togglePanel(id, true);
        } else {
            let savedW = localStorage.getItem(`pane_width_${id}`);
            if (savedW) p.el.style.width = savedW + "px";
        }
    });

    // Toggle Bindings
    if (document.getElementById("toggle-nav")) document.getElementById("toggle-nav").addEventListener("click", () => togglePanel("pane-nav"));
    if (document.getElementById("toggle-session")) document.getElementById("toggle-session").addEventListener("click", () => togglePanel("pane-session"));
    if (document.getElementById("toggle-context")) document.getElementById("toggle-context").addEventListener("click", () => togglePanel("shared-context-sidebar"));

    if ($btnExpandNav) $btnExpandNav.addEventListener("click", () => togglePanel("pane-nav"));
    if ($btnExpandSession) $btnExpandSession.addEventListener("click", () => togglePanel("pane-session"));
    if ($btnExpandContext) $btnExpandContext.addEventListener("click", () => togglePanel("shared-context-sidebar"));

    // Resizer Logic
    let isResizing = false;
    let currentResizer = null;
    let startX = 0;
    let startWidth = 0;
    let targetPanelKey = "";
    let isTargetLeft = true;

    document.addEventListener("mousedown", (e) => {
        if (e.target.classList.contains("resizer")) {
            isResizing = true;
            currentResizer = e.target;
            startX = e.clientX;
            const prev = currentResizer.getAttribute("data-prev");
            const next = currentResizer.getAttribute("data-next");

            // Priority: Resize the fixed side panels, left panel favors left resizer.
            if (prev && panels[prev]) {
                targetPanelKey = prev;
                isTargetLeft = true;
                startWidth = panels[prev].el.getBoundingClientRect().width;
            } else if (next && panels[next]) {
                targetPanelKey = next;
                isTargetLeft = false;
                startWidth = panels[next].el.getBoundingClientRect().width;
            } else {
                return;
            }

            document.body.style.cursor = "col-resize";
            e.preventDefault();
        }
    });

    document.addEventListener("mousemove", (e) => {
        if (!isResizing || !targetPanelKey) return;
        const p = panels[targetPanelKey];
        if (!p) return;

        let dx = e.clientX - startX;
        let newWidth = isTargetLeft ? startWidth + dx : startWidth - dx;
        if (newWidth < p.min) newWidth = p.min;
        if (newWidth > p.max) newWidth = p.max;

        requestAnimationFrame(() => {
            p.el.style.width = newWidth + "px";
            updateExpandButtons();
        });
    });

    document.addEventListener("mouseup", () => {
        if (isResizing && targetPanelKey) {
            isResizing = false;
            document.body.style.cursor = "";
            const finalWidth = panels[targetPanelKey].el.getBoundingClientRect().width;
            localStorage.setItem(`pane_width_${targetPanelKey}`, finalWidth);
            targetPanelKey = "";
            updateExpandButtons();
        }
    });

    // Initial expand buttons positioning after layout stabilizes
    setTimeout(updateExpandButtons, 100);
    $sessionList.addEventListener("click", async (event) => {
        const groupToggle = event.target.closest("[data-toggle-agent-group]");
        if (groupToggle) {
            const agentId = groupToggle.dataset.toggleAgentGroup;
            const nextCollapsed = !isSessionGroupCollapsed(agentId);
            setSessionGroupCollapsed(agentId, nextCollapsed);
            renderSessionList();
            return;
        }

        const deleteButton = event.target.closest("[data-delete-session]");
        if (deleteButton) {
            event.stopPropagation();
            await deleteSession(deleteButton.dataset.deleteSession);
            return;
        }

        const sessionButton = event.target.closest("[data-session-id]");
        if (!sessionButton) return;
        await switchSession(sessionButton.dataset.sessionId);
    });
    $homeView.addEventListener("click", (event) => {
        const start = event.target.closest('[data-action="start-chat"]');
        if (start) {
            openSessionPage().then(() => $input.focus());
            return;
        }

        const button = event.target.closest(".suggestion");
        if (!button) return;
        $input.value = button.dataset.prompt || "";
        autoResizeInput();
        openSessionPage().then(sendMessage);
    });

    // --- Routing & View Management ---
    const navItems = document.querySelectorAll(".nav-item[data-view]");
    const dynamicViews = document.querySelectorAll(".dynamic-view");
    const dynamicScriptLoads = new Map();

    function ensureDynamicScript(src) {
        const versionedSrc = withAssetVersion(src);
        if (dynamicScriptLoads.has(versionedSrc)) {
            return dynamicScriptLoads.get(versionedSrc);
        }
        const existing = document.querySelector(`script[src="${versionedSrc}"]`);
        if (existing) {
            const ready = Promise.resolve();
            dynamicScriptLoads.set(versionedSrc, ready);
            return ready;
        }

        const promise = new Promise((resolve, reject) => {
            const script = document.createElement("script");
            script.src = versionedSrc;
            script.async = true;
            script.onload = () => resolve();
            script.onerror = () => reject(new Error(`Failed to load script: ${versionedSrc}`));
            document.head.appendChild(script);
        });
        dynamicScriptLoads.set(versionedSrc, promise);
        return promise;
    }

    async function fetchViewContent(viewName) {
        const fileName = viewName.replace(/-/g, '_') + '.html';
        try {
            const response = await fetch(withAssetVersion('/static/' + fileName), { cache: "no-store" });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.text();
        } catch (err) {
            console.error('Failed to load view:', fileName, err);
            return `<div class="p-8 text-error flex flex-col items-center justify-center h-full">
                        <span class="material-symbols-outlined text-4xl mb-4">error</span>
                        Failed to load view: ${fileName} <br> ${err.message}
                    </div>`;
        }
    }

    async function switchView(targetView) {
        // Update nav highlights
        navItems.forEach(item => {
            const isTarget = item.dataset.view === targetView;
            if (isTarget) {
                item.classList.add("text-[#00F0FF]", "border-b-2", "border-[#00F0FF]");
                item.classList.remove("text-slate-500", "border-transparent");
                const icon = item.querySelector('span.material-symbols-outlined');
                if (icon) icon.style.fontVariationSettings = "'FILL' 1";
            } else {
                item.classList.remove("text-[#00F0FF]", "border-b-2", "border-[#00F0FF]");
                item.classList.add("text-slate-500", "border-transparent");
                const icon = item.querySelector('span.material-symbols-outlined');
                if (icon) icon.style.fontVariationSettings = "'FILL' 0";
            }
        });

        // Toggle View Visibility
        dynamicViews.forEach(view => {
            if (view.id === `view-${targetView}`) {
                view.classList.remove("hidden");
                if (targetView === "memory") {
                    view.dataset.loaded = "true";
                    renderMemoryView();
                    refreshMemoryViewData().catch((error) => {
                        console.error("Failed to refresh memory view:", error);
                    });
                } else if (targetView !== 'orchestrator' && !view.dataset.loaded) {
                    // For dynamic views, fetch content if empty
                    view.innerHTML = '<div class="p-8 flex items-center justify-center text-on-surface-variant w-full h-full"><div class="dots flex h-6 items-center px-4"><span></span><span></span><span></span></div> Loading module...</div>';
                    fetchViewContent(targetView).then(async (html) => {
                        view.innerHTML = html;
                        view.dataset.loaded = "true";

                        if (targetView === "agent-marketplace") {
                            window.renderAgentsGrid();
                            return;
                        }

                        if (targetView === "system-integrations") {
                            await ensureDynamicScript("/static/system_integrations.js");
                            if (typeof window.initSystemIntegrationsView === "function") {
                                window.initSystemIntegrationsView(view);
                            }
                        }
                        if (targetView === "scheduled-tasks") {
                            await ensureDynamicScript("/static/scheduled_tasks.js");
                            if (typeof window.initScheduledTasksView === "function") {
                                window.initScheduledTasksView(view);
                            }
                        }
                    });
                } else if (targetView === "agent-marketplace") {
                    window.renderAgentsGrid();
                } else if (targetView === "memory") {
                    refreshMemoryViewData().catch((error) => {
                        console.error("Failed to refresh memory view:", error);
                    });
                } else if (targetView === "system-integrations" && typeof window.refreshSystemIntegrationsView === "function") {
                    window.refreshSystemIntegrationsView({ preserveSelection: true }).catch((error) => {
                        console.error("Failed to refresh integrations view:", error);
                    });
                } else if (targetView === "scheduled-tasks" && typeof window.refreshScheduledTasksView === "function") {
                    window.refreshScheduledTasksView({ preserveSelection: true }).catch((error) => {
                        console.error("Failed to refresh scheduled tasks view:", error);
                    });
                }
            } else {
                view.classList.add("hidden");
            }
        });

        // Force update expand buttons layout since view switch might affect widths
        setTimeout(updateExpandButtons, 100);
    }

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            switchView(item.dataset.view);
        });
    });

    // Handle home button logo click to go back to orchestrator
    if ($logoHomeBtn) {
        // Remove the old listener binding to prevent double-binding or conflicting logic
        $logoHomeBtn.replaceWith($logoHomeBtn.cloneNode(true));
        document.getElementById("logo-home-btn").addEventListener("click", () => switchView("orchestrator"));
    }

    setSidebarTab("runs");
    bootstrap();
})();


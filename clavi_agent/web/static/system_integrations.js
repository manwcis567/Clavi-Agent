(() => {
    const state = {
        root: null,
        integrations: [],
        agents: [],
        selectedIntegrationId: "",
        bindings: [],
        routingRules: [],
        events: [],
        deliveries: [],
        selectedKind: "",
        autoRefreshLogsInterval: null,
        wechatSetup: null,
        wechatSetupPoller: null,
        integrationSavePending: false,
    };
    let wechatQrLibPromise = null;

    function getApi() {
        const api = window.ClaviAgentWorkspaceApi || {};
        if (typeof api.fetchJson !== "function") {
            throw new Error("ClaviAgentWorkspaceApi 不可用");
        }
        return api;
    }

    function $(selector) {
        return state.root ? state.root.querySelector(selector) : null;
    }

    function escapeHtml(value) {
        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function formatTime(value) {
        if (!value) {
            return "暂无记录";
        }
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) {
            return value;
        }
        return date.toLocaleString("zh-CN", { hour12: false });
    }

    function sortTimestampDesc(left, right) {
        return (Date.parse(right || "") || 0) - (Date.parse(left || "") || 0);
    }

    function integrationKindLabel(kind) {
        if (kind === "feishu") {
            return "Feishu";
        }
        if (kind === "wechat") {
            return "WeChat";
        }
        if (kind === "mock") {
            return "Mock Channel";
        }
        return kind || "未知类型";
    }

    function isFeishuKind(recordOrDraft = null) {
        return (recordOrDraft?.kind || $("#integration-kind")?.value || "mock") === "feishu";
    }

    function isWeChatKind(recordOrDraft = null) {
        return (recordOrDraft?.kind || $("#integration-kind")?.value || "mock") === "wechat";
    }

    function integrationStatusMeta(status) {
        if (status === "active") {
            return {
                label: "已启用",
                className: "border-emerald-400/20 bg-emerald-400/10 text-emerald-300",
            };
        }
        if (status === "error") {
            return {
                label: "异常",
                className: "border-error/20 bg-error/10 text-error",
            };
        }
        return {
            label: "已停用",
            className: "border-white/10 bg-white/5 text-on-surface-variant",
        };
    }

    function eventStatusMeta(status) {
        if (status === "completed") {
            return "border-emerald-400/20 bg-emerald-400/10 text-emerald-300";
        }
        if (status === "failed" || status === "rejected") {
            return "border-error/20 bg-error/10 text-error";
        }
        if (status === "bridged" || status === "routed") {
            return "border-primary/20 bg-primary/10 text-primary";
        }
        return "border-white/10 bg-white/5 text-on-surface-variant";
    }

    function deliveryStatusMeta(status) {
        if (status === "delivered") {
            return "border-emerald-400/20 bg-emerald-400/10 text-emerald-300";
        }
        if (status === "failed") {
            return "border-error/20 bg-error/10 text-error";
        }
        if (status === "retrying" || status === "sending") {
            return "border-secondary/20 bg-secondary/10 text-secondary";
        }
        return "border-white/10 bg-white/5 text-on-surface-variant";
    }

    function defaultConfigForKind(kind) {
        if (kind === "feishu") {
            return {
                app_id: "",
                connection_mode: "long_connection",
                default_agent_id: "",
                default_chat_id: "",
                default_thread_id: "",
                quick_reaction_enabled: true,
                quick_reaction_emoji_type: "DONE",
                reply_to_message: false,
                default_session_strategy: "reuse",
                outbound_retry_backoff_seconds: 1.0,
            };
        }
        if (kind === "wechat") {
            return {
                default_agent_id: "",
                setup_mode: "native_ilink",
                private_chat_only: true,
                default_session_strategy: "reuse",
            };
        }
        return {
            verify_token: "mock-token",
            default_session_strategy: "reuse",
        };
    }


    function switchView(viewName) {
        const catalogView = $("#integrations-catalog-view");
        const managementView = $("#integrations-management-view");
        if (viewName === "catalog") {
            if (catalogView) catalogView.classList.remove("hidden");
            if (managementView) managementView.classList.add("hidden");
            state.selectedKind = "";
            state.selectedIntegrationId = "";
            stopLogsRefresh();
            stopWeChatSetupPolling();
            renderSummary();
            // Re-render catalog view to reflect any changes if needed
        } else {
            if (catalogView) catalogView.classList.add("hidden");
            if (managementView) managementView.classList.remove("hidden");
        }
    }

    function switchTab(tabName) {
        const tabs = state.root.querySelectorAll(".management-tab");
        const contents = state.root.querySelectorAll(".tab-content");
        tabs.forEach(t => {
            if (t.dataset.tab === tabName) {
                t.classList.add("active", "text-primary", "border-primary");
                t.classList.remove("text-on-surface-variant", "border-transparent");
            } else {
                t.classList.remove("active", "text-primary", "border-primary");
                t.classList.add("text-on-surface-variant", "border-transparent");
            }
        });
        contents.forEach(c => {
            if (c.id === `tab-${tabName}`) {
                c.classList.remove("hidden");
                c.classList.add("block");
            } else {
                c.classList.add("hidden");
                c.classList.remove("block");
            }
        });

        if (tabName === "logs") {
            refreshSelectedLogs().catch(console.error);
            startLogsRefresh();
        } else {
            stopLogsRefresh();
        }
    }

    function startLogsRefresh() {
        if (!state.autoRefreshLogsInterval) {
            state.autoRefreshLogsInterval = setInterval(() => {
                if (state.selectedIntegrationId) {
                    refreshSelectedLogs().catch(console.error);
                }
            }, 5000);
        }
    }

    function stopLogsRefresh() {
        if (state.autoRefreshLogsInterval) {
            clearInterval(state.autoRefreshLogsInterval);
            state.autoRefreshLogsInterval = null;
        }
    }

    function stopWeChatSetupPolling() {
        if (state.wechatSetupPoller) {
            clearInterval(state.wechatSetupPoller);
            state.wechatSetupPoller = null;
        }
    }

    function openCatalogItem(kind) {
        if (!kind) return;
        state.selectedKind = kind;
        const titleEl = $("#management-kind-title");
        if (titleEl) titleEl.textContent = integrationKindLabel(kind);
        switchView("management");
        switchTab("config");

        const filtered = state.integrations.filter(item => item.kind === state.selectedKind);
        if (filtered.length > 0) {
            selectIntegration(filtered[0].id).catch(console.error);
        } else {
            resetIntegrationForm(kind);
        }
    }

    function setWorkspaceStatus(message) {
        const api = window.ClaviAgentWorkspaceApi || {};
        if (typeof api.setStatus === "function") {
            api.setStatus(message);
        }
    }

    function setFormFeedback(message, tone = "neutral") {
        const target = $("#integration-form-feedback");
        if (!target) {
            return;
        }
        const toneClasses = {
            neutral: "border-outline-variant/10 bg-black/20 text-on-surface-variant",
            success: "border-emerald-400/20 bg-emerald-400/10 text-emerald-200",
            error: "border-error/20 bg-error/10 text-error",
            info: "border-primary/20 bg-primary/10 text-primary",
        };
        target.className = `rounded-xl border px-4 py-3 text-sm ${toneClasses[tone] || toneClasses.neutral}`;
        target.textContent = message;
    }

    function setIntegrationSavePending(pending, { label = "连接中" } = {}) {
        state.integrationSavePending = Boolean(pending);
        const button = $("#integration-save-btn");
        if (!button) {
            return;
        }
        if (!button.dataset.defaultLabel) {
            button.dataset.defaultLabel = button.innerHTML;
        }
        if (state.integrationSavePending) {
            button.disabled = true;
            button.classList.add("opacity-70", "cursor-not-allowed");
            button.innerHTML = `<span class="material-symbols-outlined text-[18px]">hourglass_top</span>${escapeHtml(label)}`;
            return;
        }
        button.disabled = false;
        button.classList.remove("opacity-70", "cursor-not-allowed");
        button.innerHTML = button.dataset.defaultLabel;
    }

    function ensureSelectedIntegration() {
        if (!state.selectedIntegrationId) {
            throw new Error("请先选择或保存一个集成。");
        }
        return state.selectedIntegrationId;
    }

    function buildWebhookUrl(recordOrDraft = null) {
        const kind = recordOrDraft?.kind || $("#integration-kind")?.value || "mock";
        const integrationId = recordOrDraft?.id || $("#integration-id")?.value.trim() || "<保存后生成>";
        return `${window.location.origin}/api/integrations/${kind}/${integrationId}/webhook`;
    }

    function findCredential(record = null, credentialKey = "") {
        const normalizedKey = String(credentialKey || "").trim();
        if (!normalizedKey || !Array.isArray(record?.credentials)) {
            return null;
        }
        return record.credentials.find((item) => String(item.credential_key || "").trim() === normalizedKey) || null;
    }

    function setElementVisible(selector, visible) {
        const element = $(selector);
        if (!element) {
            return;
        }
        element.classList.toggle("hidden", !visible);
    }

    function setManagementTabVisible(tabName, visible) {
        const tabButton = state.root?.querySelector(`.management-tab[data-tab="${tabName}"]`);
        const tabContent = $(`#tab-${tabName}`);
        if (tabButton) {
            tabButton.classList.toggle("hidden", !visible);
        }
        if (!visible && tabContent) {
            tabContent.classList.add("hidden");
            tabContent.classList.remove("block");
        }
    }

    function normalizeWeChatSetupStatus(status = null) {
        return {
            integration_id: status?.integration_id || state.selectedIntegrationId || "",
            state: status?.state || "idle",
            message: status?.message || "Save a default agent, then start QR login.",
            output: status?.output || "",
            qr_text: status?.qr_text || "",
            qr_content: status?.qr_content || "",
            error: status?.error || "",
            ilink_bot_id: status?.ilink_bot_id || "",
            ilink_user_id: status?.ilink_user_id || "",
            base_url: status?.base_url || "",
            updated_at: status?.updated_at || "",
        };
    }

    function ensureWeChatQrLib() {
        if (window.qrcode) {
            return Promise.resolve(window.qrcode);
        }
        if (wechatQrLibPromise) {
            return wechatQrLibPromise;
        }
        wechatQrLibPromise = new Promise((resolve, reject) => {
            const script = document.createElement("script");
            script.src = "https://cdn.jsdelivr.net/npm/qrcode-generator@2.0.4/dist/qrcode.min.js";
            script.async = true;
            script.onload = () => {
                if (window.qrcode) {
                    resolve(window.qrcode);
                    return;
                }
                reject(new Error("QR renderer did not initialize."));
            };
            script.onerror = () => reject(new Error("Failed to load the QR renderer."));
            document.head.appendChild(script);
        });
        return wechatQrLibPromise;
    }

    function ensureWeChatSetupDom() {
        const section = $("#integration-wechat-basic-section");
        if (!section) {
            return;
        }

        const heading = section.querySelector("h3");
        if (heading) {
            heading.textContent = "WeChat Native iLink";
        }
        const intro = heading?.nextElementSibling;
        if (intro) {
            intro.textContent = "Generate the WeChat QR code directly in Clavi Agent, scan it on this page, and bind new private chats to the selected default agent.";
        }

        const agentSelect = $("#integration-wechat-default-agent-id");
        const agentLabel = agentSelect?.parentElement?.querySelector("label");
        if (agentLabel) {
            agentLabel.textContent = "Default Agent";
        }
        const agentHint = agentSelect?.nextElementSibling;
        if (agentHint) {
            agentHint.textContent = "After login succeeds, new WeChat private chats will route to this agent by default.";
        }

        const qrHeader = $("#integration-wechat-qr-empty")?.parentElement?.previousElementSibling;
        if (qrHeader?.children?.length >= 2) {
            qrHeader.children[0].textContent = "QR Code";
            qrHeader.children[1].textContent = "Native iLink";
        }

        const outputHeader = $("#integration-wechat-setup-output")?.previousElementSibling;
        if (outputHeader?.children?.length >= 2) {
            outputHeader.children[0].textContent = "Login Log";
            outputHeader.children[1].textContent = "WeChat iLink";
        }

        const qrText = $("#integration-wechat-qr-text");
        if (qrText && !$("#integration-wechat-qr-render")) {
            const qrRender = document.createElement("div");
            qrRender.id = "integration-wechat-qr-render";
            qrRender.className = "hidden min-h-[248px] items-center justify-center overflow-hidden";
            qrText.parentElement?.insertBefore(qrRender, qrText);
        }

        const legacyBotId = $("#integration-wechat-openclaw-version");
        if (legacyBotId && !$("#integration-wechat-bot-id")) {
            legacyBotId.id = "integration-wechat-bot-id";
        }

        const legacyBaseUrl = $("#integration-wechat-plugin-spec");
        if (legacyBaseUrl && !$("#integration-wechat-base-url")) {
            legacyBaseUrl.id = "integration-wechat-base-url";
        }

        const botId = $("#integration-wechat-bot-id");
        const baseUrl = $("#integration-wechat-base-url");
        if (botId && baseUrl && !$("#integration-wechat-user-id")) {
            const userId = document.createElement("div");
            userId.id = "integration-wechat-user-id";
            userId.className = baseUrl.className;
            botId.parentElement?.insertBefore(userId, baseUrl);
        }
    }

    function updateWeChatCatalogCard() {
        const card = state.root?.querySelector('[data-catalog-item="wechat"]');
        if (!card) {
            return;
        }
        const heading = card.querySelector("h3");
        if (heading) {
            heading.textContent = "WeChat";
        }
        const description = card.querySelector("p");
        if (description) {
            description.textContent = "Use native iLink QR login directly inside Clavi Agent and bind the channel to a default agent without OpenClaw.";
        }
        const badge = card.querySelector("span.inline-flex");
        if (badge) {
            badge.innerHTML = '<span class="h-1.5 w-1.5 rounded-full bg-[#48d597]"></span> Native iLink';
        }
    }

    function renderWeChatQrCode(status = null) {
        ensureWeChatSetupDom();
        const normalized = normalizeWeChatSetupStatus(status);
        const qrRender = $("#integration-wechat-qr-render");
        const qrText = $("#integration-wechat-qr-text");
        const qrEmpty = $("#integration-wechat-qr-empty");
        const qrContent = String(normalized.qr_content || "").trim();
        const fallbackText = String(normalized.qr_text || qrContent).trim();

        if (!qrRender || !qrText || !qrEmpty) {
            return;
        }
        if (!qrContent) {
            qrRender.classList.add("hidden");
            qrRender.classList.remove("flex");
            qrRender.innerHTML = "";
            qrText.classList.add("hidden");
            qrText.textContent = "";
            qrEmpty.classList.remove("hidden");
            qrEmpty.classList.add("flex");
            return;
        }

        qrEmpty.classList.add("hidden");
        qrEmpty.classList.remove("flex");
        qrText.textContent = fallbackText;
        ensureWeChatQrLib()
            .then(factory => {
                const qr = factory(0, "M");
                qr.addData(qrContent);
                qr.make();
                qrRender.innerHTML = qr.createSvgTag({ cellSize: 5, margin: 2, scalable: true });
                qrRender.classList.remove("hidden");
                qrRender.classList.add("flex");
                qrText.classList.add("hidden");
            })
            .catch(() => {
                qrRender.classList.add("hidden");
                qrRender.classList.remove("flex");
                qrRender.innerHTML = "";
                qrText.classList.remove("hidden");
            });
    }

    function weChatSetupMeta(status = null) {
        const normalized = normalizeWeChatSetupStatus(status);
        if (normalized.state === "succeeded") {
            return {
                label: "Connected",
                badgeClass: "border-emerald-400/20 bg-emerald-400/10 text-emerald-300",
                buttonLabel: "Reconnect",
            };
        }
        if (normalized.state === "failed") {
            return {
                label: "Failed",
                badgeClass: "border-error/20 bg-error/10 text-error",
                buttonLabel: "Retry Login",
            };
        }
        if (normalized.state === "waiting_scan") {
            return {
                label: "Waiting Scan",
                badgeClass: "border-[#48d597]/20 bg-[#48d597]/10 text-[#7af2be]",
                buttonLabel: "Waiting Scan",
            };
        }
        if (normalized.state === "scanned") {
            return {
                label: "Scanned",
                badgeClass: "border-[#48d597]/20 bg-[#48d597]/10 text-[#7af2be]",
                buttonLabel: "Awaiting Confirm",
            };
        }
        if (normalized.state === "running" || normalized.state === "queued") {
            return {
                label: "Connecting",
                badgeClass: "border-primary/20 bg-primary/10 text-primary",
                buttonLabel: "Connecting",
            };
        }
        return {
            label: "Idle",
            badgeClass: "border-white/10 bg-white/5 text-on-surface-variant",
            buttonLabel: "Start QR Login",
        };
    }

    function renderWeChatSetupStatus(status = null) {
        const section = $("#integration-wechat-basic-section");
        if (!section || !isWeChatKind()) {
            stopWeChatSetupPolling();
            state.wechatSetup = null;
            return;
        }
        ensureWeChatSetupDom();

        const normalized = normalizeWeChatSetupStatus(status);
        const meta = weChatSetupMeta(normalized);
        state.wechatSetup = normalized;

        const badge = $("#integration-wechat-setup-badge");
        if (badge) {
            badge.className = `inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-bold ${meta.badgeClass}`;
            badge.textContent = meta.label;
        }

        const button = $("#integration-wechat-setup-btn");
        if (button) {
            button.innerHTML = `<span class="material-symbols-outlined text-[18px]">qr_code_scanner</span>${escapeHtml(meta.buttonLabel)}`;
            button.disabled = ["queued", "running", "waiting_scan", "scanned"].includes(normalized.state);
            button.classList.toggle("opacity-60", button.disabled);
            button.classList.toggle("cursor-not-allowed", button.disabled);
        }

        const message = $("#integration-wechat-setup-message");
        if (message) {
            message.textContent = normalized.error || normalized.message;
        }

        renderWeChatQrCode(normalized);

        const output = $("#integration-wechat-setup-output");
        if (output) {
            output.textContent = normalized.output || "WeChat iLink login has not started yet.";
        }

        const botId = $("#integration-wechat-bot-id");
        if (botId) {
            botId.textContent = normalized.ilink_bot_id
                ? `Bot ID: ${normalized.ilink_bot_id}`
                : "Bot ID pending";
        }

        const userId = $("#integration-wechat-user-id");
        if (userId) {
            userId.textContent = normalized.ilink_user_id
                ? `User ID: ${normalized.ilink_user_id}`
                : "User ID pending";
        }

        const baseUrl = $("#integration-wechat-base-url");
        if (baseUrl) {
            baseUrl.textContent = normalized.base_url
                ? `Base URL: ${normalized.base_url}`
                : "Base URL pending";
        }
    }

    function setAgentBindingLock(select, hint, locked, lockedMessage, editableMessage) {
        if (!select) {
            return;
        }
        select.disabled = Boolean(locked);
        select.classList.toggle("opacity-60", Boolean(locked));
        select.classList.toggle("cursor-not-allowed", Boolean(locked));
        select.title = locked ? "默认 Agent 已在首次保存后锁定" : "";
        if (hint) {
            hint.textContent = locked ? lockedMessage : editableMessage;
        }
    }

    function fillFeishuQuickFields(record = null) {
        const config = record?.config || defaultConfigForKind("feishu");
        const appIdInput = $("#integration-feishu-app-id");
        const appSecretInput = $("#integration-feishu-app-secret");
        const appSecretStatus = $("#integration-feishu-app-secret-status");
        const defaultChatIdInput = $("#integration-feishu-default-chat-id");
        const defaultThreadIdInput = $("#integration-feishu-default-thread-id");
        const agentSelect = $("#integration-default-agent-id");
        const agentHint = agentSelect?.nextElementSibling;
        const appSecretCredential = findCredential(record, "app_secret");
        const agentLocked = Boolean(record?.id);

        if (appIdInput) {
            appIdInput.value = String(config.app_id || "");
        }
        if (appSecretInput) {
            appSecretInput.value = "";
        }
        if (defaultChatIdInput) {
            defaultChatIdInput.value = String(
                config.default_chat_id
                || config.default_target_id
                || config.target_id
                || config.receive_id
                || ""
            );
        }
        if (defaultThreadIdInput) {
            defaultThreadIdInput.value = String(
                config.default_thread_id
                || config.thread_id
                || ""
            );
        }
        if (agentSelect) {
            const defaultAgentId = String(config.default_agent_id || config.default_agent_template_id || "").trim();
            agentSelect.value = defaultAgentId || agentSelect.value || (agentSelect.options[0]?.value || "");
        }
        setAgentBindingLock(
            agentSelect,
            agentHint,
            agentLocked,
            "默认 Agent 已在首次保存后锁定；如需更换，请新建实例。",
            "所有未匹配高级路由规则的会话，将默认投递至该 Agent。"
        );
        if (appSecretStatus) {
            appSecretStatus.textContent = appSecretCredential?.masked_value
                ? `已保存：${appSecretCredential.masked_value}；留空则保持不变。`
                : "新建时必填；编辑现有集成时留空表示保持不变。";
        }
    }

    function syncHiddenFeishuConfigFields() {
        if (!isFeishuKind()) {
            return;
        }
        const textarea = $("#integration-config-json");
        if (!textarea) {
            return;
        }
        let config = {};
        try {
            config = textarea.value.trim() ? JSON.parse(textarea.value) : {};
        } catch {
            config = {};
        }
        if (!config || typeof config !== "object" || Array.isArray(config)) {
            config = {};
        }
        const nextConfig = {
            ...defaultConfigForKind("feishu"),
            ...config,
            app_id: $("#integration-feishu-app-id")?.value.trim() || "",
            default_agent_id: $("#integration-default-agent-id")?.value || "",
            default_chat_id: $("#integration-feishu-default-chat-id")?.value.trim() || "",
            default_thread_id: $("#integration-feishu-default-thread-id")?.value.trim() || "",
            connection_mode: "long_connection",
        };
        delete nextConfig.default_agent_template_id;
        delete nextConfig.app_secret;
        textarea.value = JSON.stringify(nextConfig, null, 2);
    }

    function fillWeChatQuickFields(record = null) {
        const config = record?.config || defaultConfigForKind("wechat");
        const agentSelect = $("#integration-wechat-default-agent-id");
        const agentHint = agentSelect?.nextElementSibling;
        const agentLocked = Boolean(record?.id);
        if (agentSelect) {
            const defaultAgentId = String(config.default_agent_id || "").trim();
            agentSelect.value = defaultAgentId || agentSelect.value || (agentSelect.options[0]?.value || "");
        }
        setAgentBindingLock(
            agentSelect,
            agentHint,
            agentLocked,
            "默认 Agent 已在首次保存后锁定；如需更换，请新建实例重新扫码绑定。",
            "扫码成功后，新的微信私聊会默认路由到这个 Agent。"
        );
        renderWeChatSetupStatus(record?.setup_status || record?.metadata?.wechat_setup || null);
    }

    function syncHiddenWeChatConfigFields() {
        if (!isWeChatKind()) {
            return;
        }
        const textarea = $("#integration-config-json");
        if (!textarea) {
            return;
        }
        let config = {};
        try {
            config = textarea.value.trim() ? JSON.parse(textarea.value) : {};
        } catch {
            config = {};
        }
        if (!config || typeof config !== "object" || Array.isArray(config)) {
            config = {};
        }
        const nextConfig = {
            ...defaultConfigForKind("wechat"),
            ...config,
            default_agent_id: $("#integration-wechat-default-agent-id")?.value || "",
        };
        textarea.value = JSON.stringify(nextConfig, null, 2);
    }

    function sanitizeIntegrationNamePart(value) {
        return String(value || "")
            .trim()
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "-")
            .replace(/^-+|-+$/g, "")
            .slice(0, 32);
    }

    function defaultFeishuName(config = {}) {
        const suffix = sanitizeIntegrationNamePart(config.app_id || "");
        return `feishu-${suffix || "integration"}`;
    }

    function defaultWeChatName(displayName = "") {
        const suffix = sanitizeIntegrationNamePart(displayName || "");
        return `wechat-${suffix || "channel"}`;
    }

    function defaultWeChatDisplayName() {
        return "微信扫码通道";
    }

    function defaultFeishuDisplayName(config = {}) {
        const appId = String(config.app_id || "").trim();
        if (!appId) {
            return "飞书长连接";
        }
        return `飞书 ${appId}`;
    }

    function serializeCredentialForSave(record = {}) {
        return {
            credential_key: record.credential_key || "",
            storage_kind: record.storage_kind || "env",
            secret_ref: record.secret_ref || "",
            secret_value: "",
            metadata: record.metadata || {},
        };
    }

    function buildFeishuCredentialPayload(existingRecord = null) {
        const existingCredentials = Array.isArray(existingRecord?.credentials) ? existingRecord.credentials : [];
        const preservedCredentials = existingCredentials
            .filter((item) => item.credential_key !== "app_secret")
            .map((item) => serializeCredentialForSave(item));
        const appSecretInput = $("#integration-feishu-app-secret");
        const appSecret = appSecretInput?.value.trim() || "";
        const existingAppSecret = findCredential(existingRecord, "app_secret");

        if (appSecret) {
            preservedCredentials.push({
                credential_key: "app_secret",
                storage_kind: "local_encrypted",
                secret_ref: "",
                secret_value: appSecret,
                metadata: existingAppSecret?.metadata || {},
            });
            return preservedCredentials;
        }
        if (existingAppSecret) {
            preservedCredentials.push(serializeCredentialForSave(existingAppSecret));
            return preservedCredentials;
        }
        throw new Error("App Secret 不能为空。");
    }

    function buildIntegrationConfigPayload() {
        if (isFeishuKind()) {
            syncHiddenFeishuConfigFields();
        }
        if (isWeChatKind()) {
            syncHiddenWeChatConfigFields();
        }
        const config = collectJsonConfig();
        if (!isFeishuKind() && !isWeChatKind()) {
            return config;
        }
        if (isFeishuKind() && !String(config.app_id || "").trim()) {
            throw new Error("App ID 不能为空。");
        }
        if (!String(config.default_agent_id || "").trim()) {
            throw new Error("请选择绑定的 Agent。");
        }
        return config;
    }

    function buildIntegrationCredentialsPayload(existingRecord = null) {
        if (isWeChatKind()) {
            return [];
        }
        if (!isFeishuKind()) {
            return readCredentialRows();
        }
        return buildFeishuCredentialPayload(existingRecord);
    }

    function updateIntegrationFormVisibility(record = null) {
        const isFeishu = isFeishuKind(record);
        const isWeChat = isWeChatKind(record);
        const isEditing = Boolean(record);
        const subtitle = $("#integration-panel-subtitle");

        setElementVisible("#integration-feishu-basic-section", isFeishu);
        setElementVisible("#integration-wechat-basic-section", isWeChat);
        setElementVisible("#integration-base-info-section", !isFeishu);
        setElementVisible("#integration-name-field", !isFeishu && !isWeChat);
        setElementVisible("#integration-display-name-field", !isFeishu);
        setElementVisible("#integration-tenant-field", !isFeishu && !isWeChat);
        setElementVisible("#integration-credentials-section", !isFeishu && !isWeChat);
        setElementVisible("#integration-config-section", !isFeishu && !isWeChat);
        setElementVisible("#integration-webhook-section", !isFeishu && !isWeChat);
        setManagementTabVisible("routing", !isWeChat);
        setManagementTabVisible("logs", true);

        if (!subtitle) {
            return;
        }
        if (isFeishu) {
            subtitle.textContent = isEditing
                ? "飞书接入已收敛为最小配置：App ID、App Secret 和默认 Agent。高级绑定与路由仅在需要细分会话时再配置。"
                : "只需填写 App ID、App Secret，并选择默认 Agent；保存后即可完成飞书长连接接入。";
            return;
        }
        if (isWeChat) {
            subtitle.textContent = isEditing
                ? "当前微信集成会调用腾讯官方 OpenClaw 安装器，在本页展示终端二维码并保存默认 Agent。"
                : "先选择默认 Agent，再点击“开始扫码连接”生成二维码并完成微信配对。";
            switchTab("config");
            return;
        }
        subtitle.textContent = isEditing
            ? `当前正在编辑 ${integrationKindLabel(record.kind)} 集成，可直接执行校验、启停、软删除与日志排查。`
            : "保存后即可继续配置绑定、路由和日志查看；Feishu 默认启用长连接接入。";
    }

    function resolveFeishuApiBaseUrl(recordOrDraft = null) {
        const config = currentIntegrationConfig(recordOrDraft);
        return String(config.api_base_url || "https://open.feishu.cn").trim().replace(/\/$/, "");
    }

    function currentIntegrationConfig(recordOrDraft = null) {
        if (recordOrDraft?.config && typeof recordOrDraft.config === "object") {
            return recordOrDraft.config;
        }
        if (isFeishuKind(recordOrDraft)) {
            syncHiddenFeishuConfigFields();
        }
        if (isWeChatKind(recordOrDraft)) {
            syncHiddenWeChatConfigFields();
        }
        const textarea = $("#integration-config-json");
        if (!textarea || !textarea.value.trim()) {
            return {};
        }
        try {
            return JSON.parse(textarea.value);
        } catch {
            return {};
        }
    }

    function isFeishuLongConnection(recordOrDraft = null) {
        const kind = recordOrDraft?.kind || $("#integration-kind")?.value || "mock";
        if (kind !== "feishu") {
            return false;
        }
        const config = currentIntegrationConfig(recordOrDraft);
        const rawMode = String(config.connection_mode || "").trim().toLowerCase().replace(/-/g, "_");
        return !rawMode || rawMode === "long_connection" || rawMode === "longconnection" || rawMode === "ws";
    }

    function buildIntegrationEndpointPreview(recordOrDraft = null) {
        const kind = recordOrDraft?.kind || $("#integration-kind")?.value || "mock";
        if (isFeishuLongConnection(recordOrDraft)) {
            const apiBaseUrl = resolveFeishuApiBaseUrl(recordOrDraft);
            return {
                title: "长连接接入",
                hint: "当前为 Feishu 长连接模式，不需要配置本地回调地址。服务启用后会使用 app_id / app_secret 主动连接飞书开放平台；下面显示的是官方接入域名。",
                value: `${apiBaseUrl}  (SDK 将动态申请长连接地址)`,
                copyValue: apiBaseUrl,
                copyLabel: "复制域名",
            };
        }
        return {
            title: kind === "feishu" ? "Webhook 回调地址" : "接入地址",
            hint: "地址由系统按渠道类型自动生成。Mock 与 webhook 模式可直接复制给外部平台使用。",
            value: buildWebhookUrl(recordOrDraft),
            copyValue: buildWebhookUrl(recordOrDraft),
            copyLabel: "复制地址",
        };
    }

    function updateWebhookPreview(record = null) {
        const input = $("#integration-webhook-url");
        const title = $("#integration-endpoint-title");
        const hint = $("#integration-endpoint-hint");
        const copyLabel = $("#integration-copy-webhook-label");
        const preview = buildIntegrationEndpointPreview(record);
        if (title) {
            title.textContent = preview.title;
        }
        if (hint) {
            hint.textContent = preview.hint;
        }
        if (copyLabel) {
            copyLabel.textContent = preview.copyLabel;
        }
        if (input) {
            input.value = preview.value;
            input.dataset.copyValue = preview.copyValue;
        }
    }

    function ensureConfigTemplate(force = false) {
        const textarea = $("#integration-config-json");
        const hiddenId = $("#integration-id");
        const kind = $("#integration-kind")?.value || "mock";
        if (!textarea) {
            return;
        }
        if (!force && hiddenId?.value) {
            return;
        }
        if (!force && textarea.value.trim()) {
            return;
        }
        textarea.value = JSON.stringify(defaultConfigForKind(kind), null, 2);
    }

    function renderAgentOptions() {
        const selects = [
            $("#integration-default-agent-id"),
            $("#integration-wechat-default-agent-id"),
            $("#binding-agent-id"),
            $("#routing-agent-id"),
        ];
        const options = state.agents.length
            ? state.agents
                .map((agent) => `<option value="${escapeHtml(agent.id)}">${escapeHtml(agent.name || agent.id)}</option>`)
                .join("")
            : `<option value="">暂无可用 Agent</option>`;
        selects.forEach((select) => {
            if (!select) {
                return;
            }
            const currentValue = select.value;
            select.innerHTML = options;
            if (currentValue && Array.from(select.options).some((item) => item.value === currentValue)) {
                select.value = currentValue;
            }
        });
    }

    function renderSummary() {
        const activeCount = state.integrations.filter((item) => item.status === "active").length;
        const errorCount = state.integrations.filter((item) => item.status === "error").length;
        const latestActivity = state.integrations
            .flatMap((item) => [item.last_event_at, item.last_delivery_at].filter(Boolean))
            .sort(sortTimestampDesc)[0] || "";

        if ($("#summary-active-count")) {
            $("#summary-active-count").textContent = String(activeCount);
        }
        if ($("#summary-error-count")) {
            $("#summary-error-count").textContent = String(errorCount);
        }
        if ($("#summary-last-activity")) {
            $("#summary-last-activity").textContent = latestActivity ? formatTime(latestActivity) : "暂无记录";
        }
    }

    function renderIntegrationsList() {
        const container = $("#integrations-list");
        if (!container) {
            return;
        }
        const filtered = state.integrations.filter(item => item.kind === state.selectedKind);
        if (!filtered.length) {
            container.innerHTML = `
                <div class="rounded-2xl border border-dashed border-outline-variant/20 bg-black/20 px-4 py-8 text-center text-[11px] leading-relaxed text-on-surface-variant">
                    未发现 ${escapeHtml(integrationKindLabel(state.selectedKind))} 实例。<br>点击上方立即新建。
                </div>
            `;
            return;
        }

        container.innerHTML = filtered
            .map((integration) => {
                const meta = integrationStatusMeta(integration.status);
                const isActive = integration.id === state.selectedIntegrationId;
                return `
                    <div role="button" tabindex="0"
                        data-integration-select="${escapeHtml(integration.id)}"
                        class="w-full rounded-[16px] border px-4 py-4 text-left transition-all relative overflow-hidden ${isActive
                        ? "border-primary/40 bg-primary/10 shadow-[0_0_18px_rgba(0,240,255,0.08)]"
                        : "border-white/5 bg-black/20 hover:border-primary/20 hover:bg-black/40"}">
                        ${isActive ? '<div class="absolute left-0 top-0 bottom-0 w-1 bg-primary"></div>' : ''}
                        <div class="flex items-start justify-between gap-3">
                            <div class="min-w-0">
                                <div class="text-sm font-bold text-white truncate">${escapeHtml(integration.display_name || integration.name)}</div>
                            </div>
                            <div class="shrink-0 flex items-center gap-2">
                                <span class="inline-flex items-center rounded-full border px-2 py-0.5 text-[9px] font-bold ${meta.className}">
                                    ${escapeHtml(meta.label)}
                                </span>
                                <button
                                    type="button"
                                    data-action="delete-integration-inline"
                                    data-integration-id="${escapeHtml(integration.id)}"
                                    data-integration-name="${escapeHtml(integration.display_name || integration.name)}"
                                    class="inline-flex h-7 w-7 items-center justify-center rounded-lg border border-error/20 bg-error/10 text-error transition-colors hover:bg-error/20"
                                    title="删除实例"
                                >
                                    <span class="material-symbols-outlined text-[16px]">delete</span>
                                </button>
                            </div>
                        </div>
                    </div>
                `;
            })
            .join("");
    }

    function updateIntegrationActionButtons(record = null) {
        const enableButton = $("#integration-enable-btn");
        const disableButton = $("#integration-disable-btn");
        const deleteButton = $("#integration-delete-btn");
        const isEditing = Boolean(record?.id);
        const canToggle = isEditing && !record?.deleted;

        if (enableButton) {
            enableButton.classList.toggle("hidden", !(canToggle && record?.status !== "active"));
        }
        if (disableButton) {
            disableButton.classList.toggle("hidden", !(canToggle && record?.status === "active"));
        }
        if (deleteButton) {
            deleteButton.classList.toggle("hidden", !isEditing);
        }
    }

    function credentialRowTemplate(record = {}) {
        const storageKind = record.storage_kind || "env";
        const valuePlaceholder = storageKind === "local_encrypted"
            ? "如需更新密文，请重新输入"
            : "仅 local_encrypted 使用此字段";
        return `
            <div data-credential-row class="rounded-2xl border border-outline-variant/10 bg-surface-container-high px-4 py-4">
                <div class="grid grid-cols-1 gap-3 md:grid-cols-12">
                    <div class="md:col-span-3">
                        <label class="mb-2 block text-[11px] font-bold uppercase tracking-[0.16em] text-on-surface-variant">凭证键</label>
                        <input data-field="credential_key"
                            class="w-full rounded-xl border border-outline-variant/20 bg-surface-container-highest px-3 py-2.5 text-sm text-white outline-none transition-colors focus:border-primary/50"
                            value="${escapeHtml(record.credential_key || "")}"
                            placeholder="例如 app_secret">
                    </div>
                    <div class="md:col-span-3">
                        <label class="mb-2 block text-[11px] font-bold uppercase tracking-[0.16em] text-on-surface-variant">存储方式</label>
                        <select data-field="storage_kind"
                            class="w-full rounded-xl border border-outline-variant/20 bg-surface-container-highest px-3 py-2.5 text-sm text-white outline-none transition-colors focus:border-primary/50">
                            <option value="env" ${storageKind === "env" ? "selected" : ""}>env</option>
                            <option value="external_ref" ${storageKind === "external_ref" ? "selected" : ""}>external_ref</option>
                            <option value="local_encrypted" ${storageKind === "local_encrypted" ? "selected" : ""}>local_encrypted</option>
                        </select>
                    </div>
                    <div class="md:col-span-4">
                        <label class="mb-2 block text-[11px] font-bold uppercase tracking-[0.16em] text-on-surface-variant">引用 / 密文</label>
                        <input data-field="secret_ref"
                            class="w-full rounded-xl border border-outline-variant/20 bg-surface-container-highest px-3 py-2.5 text-sm text-white outline-none transition-colors focus:border-primary/50"
                            value="${escapeHtml(record.secret_ref || "")}"
                            placeholder="例如 FEISHU_APP_SECRET">
                        <input data-field="secret_value"
                            class="mt-2 w-full rounded-xl border border-outline-variant/20 bg-surface-container-highest px-3 py-2.5 text-sm text-white outline-none transition-colors focus:border-primary/50"
                            value=""
                            placeholder="${escapeHtml(valuePlaceholder)}">
                        <div class="mt-2 text-[11px] leading-relaxed text-on-surface-variant">
                            已保存：${escapeHtml(record.masked_value || "尚未保存")}
                        </div>
                    </div>
                    <div class="md:col-span-2 flex items-end justify-end">
                        <button type="button" data-action="remove-credential"
                            class="inline-flex items-center gap-2 rounded-xl border border-error/20 bg-error/10 px-3 py-2 text-xs font-bold text-error transition-colors hover:bg-error/15">
                            <span class="material-symbols-outlined text-[16px]">delete</span>
                            删除
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    function renderCredentialRows(records = []) {
        const container = $("#integration-credential-rows");
        if (!container) {
            return;
        }
        const normalized = Array.isArray(records) && records.length ? records : [{}];
        container.innerHTML = normalized.map((item) => credentialRowTemplate(item)).join("");
    }

    function readCredentialRows() {
        const rows = Array.from(state.root.querySelectorAll("[data-credential-row]"));
        return rows.reduce((items, row) => {
            const credentialKey = row.querySelector('[data-field="credential_key"]')?.value.trim() || "";
            const storageKind = row.querySelector('[data-field="storage_kind"]')?.value || "env";
            const secretRef = row.querySelector('[data-field="secret_ref"]')?.value.trim() || "";
            const secretValue = row.querySelector('[data-field="secret_value"]')?.value || "";
            if (!credentialKey && !secretRef && !secretValue) {
                return items;
            }
            if (!credentialKey) {
                throw new Error("凭证键不能为空。");
            }
            items.push({
                credential_key: credentialKey,
                storage_kind: storageKind,
                secret_ref: secretRef,
                secret_value: secretValue,
            });
            return items;
        }, []);
    }

    function getSelectedIntegration() {
        return state.integrations.find((item) => item.id === state.selectedIntegrationId) || null;
    }

    function fillIntegrationForm(record = null) {
        const isEditing = Boolean(record);
        const nameInput = $("#integration-name");
        const displayNameInput = $("#integration-display-name");
        const kindInput = $("#integration-kind");
        const tenantInput = $("#integration-tenant-id");
        const idInput = $("#integration-id");
        const title = $("#integration-panel-title");
        const subtitle = $("#integration-panel-subtitle");
        const status = $("#integration-current-status");
        const configTextarea = $("#integration-config-json");
        if (!nameInput || !displayNameInput || !kindInput || !tenantInput || !idInput || !title || !subtitle || !status || !configTextarea) {
            return;
        }

        idInput.value = record?.id || "";
        nameInput.value = record?.name || "";
        displayNameInput.value = record?.display_name || "";
        kindInput.value = record?.kind || kindInput.value || "mock";
        tenantInput.value = record?.tenant_id || "";
        configTextarea.value = JSON.stringify(record?.config || defaultConfigForKind(kindInput.value), null, 2);
        title.textContent = isEditing ? (record.display_name || record.name) : "新建集成";

        const meta = integrationStatusMeta(record?.status || "disabled");
        status.className = `inline-flex items-center rounded-full border px-3 py-1.5 text-[11px] font-bold ${isEditing ? meta.className : "border-outline-variant/20 bg-black/20 text-on-surface-variant"}`;
        status.textContent = isEditing ? meta.label : "尚未保存";
        updateIntegrationActionButtons(record);

        renderCredentialRows(record?.credentials || []);
        fillFeishuQuickFields(record);
        fillWeChatQuickFields(record);
        syncHiddenFeishuConfigFields();
        syncHiddenWeChatConfigFields();
        updateIntegrationFormVisibility(record);
        updateWebhookPreview(record);
    }

    function resetIntegrationForm(forceKind = null) {
        state.selectedIntegrationId = "";
        stopWeChatSetupPolling();
        state.wechatSetup = null;
        const kindInput = $("#integration-kind");
        if (kindInput && forceKind && typeof forceKind === "string") {
            kindInput.value = forceKind;
        } else if (kindInput && state.selectedKind) {
            kindInput.value = state.selectedKind;
        }
        fillIntegrationForm(null);
        renderBindingsList();
        renderRoutingRulesList();
        renderEventsList();
        renderDeliveriesList();
        ensureConfigTemplate(true);
        renderIntegrationsList();
        setFormFeedback("已切换到新建模式。填写并保存后即可继续配置绑定和日志。", "info");
        switchTab("config");
    }

    function fillBindingForm(binding = null) {
        $("#binding-id").value = binding?.id || "";
        $("#binding-tenant-id").value = binding?.tenant_id || "";
        $("#binding-chat-id").value = binding?.chat_id || "";
        $("#binding-thread-id").value = binding?.thread_id || "";
        $("#binding-scope").value = binding?.binding_scope || "chat";
        $("#binding-agent-id").value = binding?.agent_id || ($("#binding-agent-id").options[0]?.value || "");
        $("#binding-refresh-session").checked = false;
    }

    function resetBindingForm() {
        fillBindingForm(null);
    }

    function fillRoutingForm(rule = null) {
        $("#routing-rule-id").value = rule?.id || "";
        $("#routing-priority").value = String(rule?.priority ?? 100);
        $("#routing-match-type").value = rule?.match_type || "integration_id";
        $("#routing-match-value").value = rule?.match_value || "";
        $("#routing-agent-id").value = rule?.agent_id || ($("#routing-agent-id").options[0]?.value || "");
        $("#routing-session-strategy").value = rule?.session_strategy || "reuse";
        $("#routing-enabled").checked = rule?.enabled ?? true;
    }

    function resetRoutingForm() {
        fillRoutingForm(null);
    }

    function collectJsonConfig() {
        const raw = $("#integration-config-json")?.value.trim() || "";
        if (!raw) {
            return {};
        }
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
            throw new Error("渠道配置 JSON 必须是对象。");
        }
        return parsed;
    }

    function renderBindingsList() {
        const container = $("#bindings-list");
        if (!container) {
            return;
        }
        if (!state.selectedIntegrationId) {
            container.innerHTML = `
                <div class="rounded-2xl border border-dashed border-outline-variant/20 bg-surface-container px-4 py-6 text-sm text-on-surface-variant">
                    请先保存或选择一个集成，再为它配置会话绑定。
                </div>
            `;
            return;
        }
        if (!state.bindings.length) {
            container.innerHTML = `
                <div class="rounded-2xl border border-dashed border-outline-variant/20 bg-surface-container px-4 py-6 text-sm text-on-surface-variant">
                    当前没有绑定记录。建议先为一个群聊或线程建立显式绑定，再进行消息联调。
                </div>
            `;
            return;
        }

        container.innerHTML = state.bindings.map((binding) => `
            <div class="rounded-2xl border border-outline-variant/10 bg-surface-container px-4 py-4">
                <div class="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                    <div class="min-w-0">
                        <div class="flex flex-wrap items-center gap-2">
                            <div class="text-sm font-bold text-white">${escapeHtml(binding.chat_id || "未填写 Chat ID")}</div>
                            <span class="inline-flex items-center rounded-full border px-2.5 py-1 text-[10px] font-bold ${binding.enabled ? "border-emerald-400/20 bg-emerald-400/10 text-emerald-300" : "border-white/10 bg-white/5 text-on-surface-variant"}">
                                ${binding.enabled ? "启用中" : "已停用"}
                            </span>
                        </div>
                        <div class="mt-2 text-xs leading-relaxed text-on-surface-variant">
                            Agent：${escapeHtml(binding.agent_id)} · Scope：${escapeHtml(binding.binding_scope)} · Session：${escapeHtml(binding.session_id)}
                        </div>
                        <div class="mt-1 text-xs leading-relaxed text-on-surface-variant">
                            Thread：${escapeHtml(binding.thread_id || "__root__")} · Tenant：${escapeHtml(binding.tenant_id || "未填写")}
                        </div>
                    </div>
                    <div class="flex flex-wrap items-center gap-2">
                        <button type="button" data-action="edit-binding" data-binding-id="${escapeHtml(binding.id)}"
                            class="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs font-bold text-on-surface transition-colors hover:border-primary/40 hover:text-primary">
                            <span class="material-symbols-outlined text-[16px]">edit</span>
                            编辑
                        </button>
                        <button type="button" data-action="toggle-binding" data-binding-id="${escapeHtml(binding.id)}" data-enabled="${binding.enabled ? "true" : "false"}"
                            class="inline-flex items-center gap-2 rounded-xl border ${binding.enabled ? "border-white/10 bg-white/5 text-on-surface" : "border-emerald-400/20 bg-emerald-400/10 text-emerald-300"} px-3 py-2 text-xs font-bold transition-colors">
                            <span class="material-symbols-outlined text-[16px]">${binding.enabled ? "pause_circle" : "play_circle"}</span>
                            ${binding.enabled ? "停用" : "启用"}
                        </button>
                        <button type="button" data-action="delete-binding" data-binding-id="${escapeHtml(binding.id)}"
                            class="inline-flex items-center gap-2 rounded-xl border border-error/20 bg-error/10 px-3 py-2 text-xs font-bold text-error transition-colors hover:bg-error/15">
                            <span class="material-symbols-outlined text-[16px]">delete</span>
                            删除
                        </button>
                    </div>
                </div>
            </div>
        `).join("");
    }

    function renderRoutingRulesList() {
        const container = $("#routing-rules-list");
        if (!container) {
            return;
        }
        if (!state.selectedIntegrationId) {
            container.innerHTML = `
                <div class="rounded-2xl border border-dashed border-outline-variant/20 bg-surface-container px-4 py-6 text-sm text-on-surface-variant">
                    请选择一个集成后再配置路由规则。
                </div>
            `;
            return;
        }
        if (!state.routingRules.length) {
            container.innerHTML = `
                <div class="rounded-2xl border border-dashed border-outline-variant/20 bg-surface-container px-4 py-6 text-sm text-on-surface-variant">
                    当前没有路由规则。没有显式绑定时，消息会走默认 Agent 回退逻辑。
                </div>
            `;
            return;
        }

        container.innerHTML = state.routingRules.map((rule) => `
            <div class="rounded-2xl border border-outline-variant/10 bg-surface-container px-4 py-4">
                <div class="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                    <div class="min-w-0">
                        <div class="flex flex-wrap items-center gap-2">
                            <div class="text-sm font-bold text-white">${escapeHtml(rule.match_type)} = ${escapeHtml(rule.match_value)}</div>
                            <span class="inline-flex items-center rounded-full border px-2.5 py-1 text-[10px] font-bold ${rule.enabled ? "border-secondary/20 bg-secondary/10 text-secondary" : "border-white/10 bg-white/5 text-on-surface-variant"}">
                                ${rule.enabled ? "启用中" : "已停用"}
                            </span>
                        </div>
                        <div class="mt-2 text-xs leading-relaxed text-on-surface-variant">
                            Agent：${escapeHtml(rule.agent_id)} · 优先级：${rule.priority} · Session Strategy：${escapeHtml(rule.session_strategy)}
                        </div>
                    </div>
                    <div class="flex flex-wrap items-center gap-2">
                        <button type="button" data-action="edit-rule" data-rule-id="${escapeHtml(rule.id)}"
                            class="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs font-bold text-on-surface transition-colors hover:border-primary/40 hover:text-primary">
                            <span class="material-symbols-outlined text-[16px]">edit</span>
                            编辑
                        </button>
                        <button type="button" data-action="toggle-rule" data-rule-id="${escapeHtml(rule.id)}" data-enabled="${rule.enabled ? "true" : "false"}"
                            class="inline-flex items-center gap-2 rounded-xl border ${rule.enabled ? "border-white/10 bg-white/5 text-on-surface" : "border-secondary/20 bg-secondary/10 text-secondary"} px-3 py-2 text-xs font-bold transition-colors">
                            <span class="material-symbols-outlined text-[16px]">${rule.enabled ? "pause_circle" : "play_circle"}</span>
                            ${rule.enabled ? "停用" : "启用"}
                        </button>
                        <button type="button" data-action="delete-rule" data-rule-id="${escapeHtml(rule.id)}"
                            class="inline-flex items-center gap-2 rounded-xl border border-error/20 bg-error/10 px-3 py-2 text-xs font-bold text-error transition-colors hover:bg-error/15">
                            <span class="material-symbols-outlined text-[16px]">delete</span>
                            删除
                        </button>
                    </div>
                </div>
            </div>
        `).join("");
    }

    function renderEventsList() {
        const container = $("#integration-events-list");
        if (!container) {
            return;
        }
        if (!state.selectedIntegrationId) {
            container.innerHTML = `
                <div class="rounded-2xl border border-dashed border-outline-variant/20 bg-surface-container px-4 py-6 text-sm text-on-surface-variant">
                    选择一个集成后，这里会显示最近入站事件。
                </div>
            `;
            return;
        }
        if (!state.events.length) {
            container.innerHTML = `
                <div class="rounded-2xl border border-dashed border-outline-variant/20 bg-surface-container px-4 py-6 text-sm text-on-surface-variant">
                    暂无入站事件。启用集成并发送一条测试消息后，这里会自动出现审计记录。
                </div>
            `;
            return;
        }

        container.innerHTML = state.events.map((event) => `
            <div class="rounded-2xl border border-outline-variant/10 bg-surface-container px-4 py-4">
                <div class="flex flex-wrap items-center gap-2">
                    <div class="text-sm font-bold text-white">${escapeHtml(event.event_type)}</div>
                    <span class="inline-flex items-center rounded-full border px-2.5 py-1 text-[10px] font-bold ${eventStatusMeta(event.normalized_status)}">
                        ${escapeHtml(event.normalized_status)}
                    </span>
                </div>
                <div class="mt-2 text-xs leading-relaxed text-on-surface-variant">
                    收到时间：${escapeHtml(formatTime(event.received_at))} · Chat：${escapeHtml(event.provider_chat_id || "未记录")} · Message：${escapeHtml(event.provider_message_id || "未记录")}
                </div>
                <div class="mt-2 text-xs leading-relaxed ${event.normalized_error ? "text-error" : "text-on-surface-variant"}">
                    ${escapeHtml(event.normalized_error || "当前事件无错误信息。")}
                </div>
            </div>
        `).join("");
    }

    function renderDeliveriesList() {
        const container = $("#integration-deliveries-list");
        if (!container) {
            return;
        }
        if (!state.selectedIntegrationId) {
            container.innerHTML = `
                <div class="rounded-2xl border border-dashed border-outline-variant/20 bg-surface-container px-4 py-6 text-sm text-on-surface-variant">
                    选择一个集成后，这里会显示最近渠道回写记录。
                </div>
            `;
            return;
        }
        if (!state.deliveries.length) {
            container.innerHTML = `
                <div class="rounded-2xl border border-dashed border-outline-variant/20 bg-surface-container px-4 py-6 text-sm text-on-surface-variant">
                    暂无回写记录。只有在消息成功路由到 Agent 并产生回复后，这里才会出现投递日志。
                </div>
            `;
            return;
        }

        container.innerHTML = state.deliveries.map((delivery) => {
            const attemptSummary = Array.isArray(delivery.attempts) && delivery.attempts.length
                ? delivery.attempts.map((attempt) => `${attempt.attempt_number}:${attempt.status}`).join(" / ")
                : "暂无尝试明细";
            return `
                <div class="rounded-2xl border border-outline-variant/10 bg-surface-container px-4 py-4">
                    <div class="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                        <div class="min-w-0">
                            <div class="flex flex-wrap items-center gap-2">
                                <div class="text-sm font-bold text-white">${escapeHtml(delivery.delivery_type)}</div>
                                <span class="inline-flex items-center rounded-full border px-2.5 py-1 text-[10px] font-bold ${deliveryStatusMeta(delivery.status)}">
                                    ${escapeHtml(delivery.status)}
                                </span>
                            </div>
                            <div class="mt-2 text-xs leading-relaxed text-on-surface-variant">
                                Run：${escapeHtml(delivery.run_id)} · 尝试次数：${delivery.attempt_count} · 最近尝试：${escapeHtml(formatTime(delivery.last_attempt_at || delivery.updated_at))}
                            </div>
                            <div class="mt-2 text-xs leading-relaxed text-on-surface-variant">
                                Attempt 明细：${escapeHtml(attemptSummary)}
                            </div>
                            <div class="mt-2 text-xs leading-relaxed ${delivery.error_summary ? "text-error" : "text-on-surface-variant"}">
                                ${escapeHtml(delivery.error_summary || "当前投递没有错误摘要。")}
                            </div>
                        </div>
                        <div class="flex flex-wrap items-center gap-2">
                            ${(delivery.status === "failed" || delivery.status === "retrying") ? `
                                <button type="button" data-action="retry-delivery" data-delivery-id="${escapeHtml(delivery.id)}"
                                    class="inline-flex items-center gap-2 rounded-xl border border-secondary/20 bg-secondary/10 px-3 py-2 text-xs font-bold text-secondary transition-colors hover:bg-secondary/15">
                                    <span class="material-symbols-outlined text-[16px]">replay</span>
                                    手动重试
                                </button>
                            ` : ""}
                        </div>
                    </div>
                </div>
            `;
        }).join("");
    }

    async function refreshAgentsAndIntegrations({ preserveSelection = true } = {}) {
        const api = getApi();
        const [agents, integrations] = await Promise.all([
            api.fetchJson("/api/agents").catch(() => []),
            api.fetchJson("/api/integrations"),
        ]);
        state.agents = Array.isArray(agents) ? agents : [];
        state.integrations = Array.isArray(integrations) ? integrations : [];
        renderAgentOptions();
        renderSummary();

        const previousId = preserveSelection ? state.selectedIntegrationId : "";
        if (previousId && state.integrations.some((item) => item.id === previousId)) {
            state.selectedIntegrationId = previousId;
        } else if (!preserveSelection) {
            state.selectedIntegrationId = "";
        } else if (state.selectedIntegrationId && !state.integrations.some((item) => item.id === state.selectedIntegrationId)) {
            state.selectedIntegrationId = "";
        }

        renderIntegrationsList();
    }

    async function loadSelectedIntegrationResources() {
        if (!state.selectedIntegrationId) {
            state.bindings = [];
            state.routingRules = [];
            state.events = [];
            state.deliveries = [];
            renderBindingsList();
            renderRoutingRulesList();
            renderEventsList();
            renderDeliveriesList();
            return;
        }

        const api = getApi();
        const integrationId = state.selectedIntegrationId;
        const eventsStatus = $("#events-status-filter")?.value || "";
        const deliveriesStatus = $("#deliveries-status-filter")?.value || "";
        const [bindings, rules, events, deliveries] = await Promise.all([
            api.fetchJson(`/api/integrations/${encodeURIComponent(integrationId)}/bindings`),
            api.fetchJson(`/api/integrations/${encodeURIComponent(integrationId)}/routing-rules`),
            api.fetchJson(`/api/integrations/${encodeURIComponent(integrationId)}/events?limit=12${eventsStatus ? `&status=${encodeURIComponent(eventsStatus)}` : ""}`),
            api.fetchJson(`/api/integrations/${encodeURIComponent(integrationId)}/deliveries?limit=12${deliveriesStatus ? `&status=${encodeURIComponent(deliveriesStatus)}` : ""}`),
        ]);

        state.bindings = Array.isArray(bindings) ? bindings : [];
        state.routingRules = Array.isArray(rules) ? rules : [];
        state.events = Array.isArray(events) ? events : [];
        state.deliveries = Array.isArray(deliveries) ? deliveries : [];
        renderBindingsList();
        renderRoutingRulesList();
        renderEventsList();
        renderDeliveriesList();
    }

    async function selectIntegration(integrationId) {
        state.selectedIntegrationId = integrationId;
        const record = getSelectedIntegration();
        stopWeChatSetupPolling();
        fillIntegrationForm(record);
        renderIntegrationsList();
        resetBindingForm();
        resetRoutingForm();
        await loadSelectedIntegrationResources();
        if (isWeChatKind(record)) {
            renderWeChatSetupStatus(record?.setup_status || null);
            if (record?.setup_status && ["queued", "running", "waiting_scan"].includes(record.setup_status.state)) {
                await pollWeChatSetupStatus();
            }
        }
        setFormFeedback(`已切换到集成：${record.display_name || record.name}`, "info");
    }

    async function persistIntegration({ feedback = true } = {}) {
        if (state.integrationSavePending) {
            throw new Error("配置保存中，请稍候。");
        }
        const api = getApi();
        setIntegrationSavePending(true, { label: "连接中" });
        try {
            const integrationId = $("#integration-id").value.trim();
            const currentRecord = getSelectedIntegration();
            const existingRecord = currentRecord && currentRecord.id === integrationId && currentRecord.kind === $("#integration-kind").value
                ? currentRecord
                : null;
            const config = buildIntegrationConfigPayload();
            const wechatDisplayName = ($("#integration-display-name").value.trim() || existingRecord?.display_name || defaultWeChatDisplayName()).trim();
            const payload = {
                name: isFeishuKind()
                    ? (existingRecord?.name || defaultFeishuName(config))
                    : isWeChatKind()
                        ? (existingRecord?.name || defaultWeChatName(wechatDisplayName))
                        : $("#integration-name").value.trim(),
                display_name: isFeishuKind()
                    ? (existingRecord?.display_name || defaultFeishuDisplayName(config))
                    : isWeChatKind()
                        ? wechatDisplayName
                        : $("#integration-display-name").value.trim(),
                kind: $("#integration-kind").value,
                tenant_id: (isFeishuKind() || isWeChatKind()) ? "" : $("#integration-tenant-id").value.trim(),
                config,
                credentials: buildIntegrationCredentialsPayload(existingRecord),
                metadata: existingRecord?.metadata || {},
                enabled: integrationId ? undefined : isFeishuKind(),
            };
            if (!payload.name) {
                throw new Error("名称不能为空。");
            }

            const method = integrationId ? "PATCH" : "POST";
            const url = integrationId
                ? `/api/integrations/${encodeURIComponent(integrationId)}`
                : "/api/integrations";
            const saved = await api.fetchJson(url, {
                method,
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            state.selectedIntegrationId = saved.id;
            await refreshSystemIntegrationsView({ preserveSelection: true });
            await selectIntegration(saved.id);
            if (feedback) {
                setFormFeedback("集成配置已保存。", "success");
                setWorkspaceStatus(`已保存集成：${saved.display_name || saved.name}`);
            }
            return saved;
        } finally {
            setIntegrationSavePending(false);
        }
    }

    async function saveIntegration(event) {
        event.preventDefault();
        try {
            await persistIntegration({ feedback: true });
        } catch (error) {
            console.error(error);
            setFormFeedback(error.message || "保存失败。", "error");
            setWorkspaceStatus("集成保存失败");
        }
    }

    async function pollWeChatSetupStatus(immediate = false) {
        const api = getApi();
        const integrationId = state.selectedIntegrationId;
        if (!integrationId || !isWeChatKind()) {
            stopWeChatSetupPolling();
            return;
        }

        const runPoll = async () => {
            try {
                const status = await api.fetchJson(`/api/integrations/${encodeURIComponent(integrationId)}/wechat/setup`);
                renderWeChatSetupStatus(status);
                const record = getSelectedIntegration();
                if (record) {
                    record.setup_status = status;
                }
                if (!["queued", "running", "waiting_scan"].includes(status.state)) {
                    stopWeChatSetupPolling();
                }
            } catch (error) {
                console.error(error);
                stopWeChatSetupPolling();
            }
        };

        if (immediate) {
            await runPoll();
        }
        if (!state.wechatSetupPoller) {
            state.wechatSetupPoller = setInterval(() => {
                runPoll().catch(console.error);
            }, 1500);
        }
    }

    async function startWeChatSetup() {
        const api = getApi();
        try {
            stopWeChatSetupPolling();
            const saved = await persistIntegration({ feedback: false });
            const status = await api.fetchJson(`/api/integrations/${encodeURIComponent(saved.id)}/wechat/setup`, {
                method: "POST",
            });
            renderWeChatSetupStatus(status);
            setFormFeedback("微信扫码任务已启动，等待二维码生成。", "info");
            setWorkspaceStatus(`已启动微信扫码：${saved.display_name || saved.name}`);
            await pollWeChatSetupStatus(true);
        } catch (error) {
            console.error(error);
            setFormFeedback(error.message || "微信扫码启动失败。", "error");
        }
    }

    async function verifySelectedIntegration() {
        const api = getApi();
        try {
            const integrationId = ensureSelectedIntegration();
            const result = await api.fetchJson(`/api/integrations/${encodeURIComponent(integrationId)}/verify`, {
                method: "POST",
            });
            await refreshSystemIntegrationsView({ preserveSelection: true });
            await selectIntegration(result.integration.id);
            setFormFeedback(result.message, result.success ? "success" : "error");
            setWorkspaceStatus(`已完成配置校验：${result.integration.display_name || result.integration.name}`);
        } catch (error) {
            console.error(error);
            setFormFeedback(error.message || "配置校验失败。", "error");
        }
    }

    async function updateSelectedIntegrationStatus(nextStatus) {
        const api = getApi();
        try {
            const integrationId = ensureSelectedIntegration();
            const endpoint = nextStatus === "active" ? "enable" : "disable";
            const record = await api.fetchJson(`/api/integrations/${encodeURIComponent(integrationId)}/${endpoint}`, {
                method: "POST",
            });
            await refreshSystemIntegrationsView({ preserveSelection: true });
            await selectIntegration(record.id);
            setFormFeedback(nextStatus === "active" ? "集成已启用。" : "集成已停用。", "success");
            setWorkspaceStatus(`${record.display_name || record.name} 已${nextStatus === "active" ? "启用" : "停用"}`);
        } catch (error) {
            console.error(error);
            setFormFeedback(error.message || "状态更新失败。", "error");
        }
    }

    async function deleteSelectedIntegration() {
        const api = getApi();
        try {
            const integrationId = ensureSelectedIntegration();
            const record = getSelectedIntegration();
            const integrationName = record?.display_name || record?.name || integrationId;
            if (!window.confirm(`确认删除实例“${integrationName}”吗？此操作会停用相关绑定和路由规则。`)) {
                setFormFeedback("已取消删除。", "info");
                return;
            }
            await api.fetchJson(`/api/integrations/${encodeURIComponent(integrationId)}`, {
                method: "DELETE",
            });
            await refreshSystemIntegrationsView({ preserveSelection: false });
            resetIntegrationForm();
            setWorkspaceStatus("集成已软删除");
        } catch (error) {
            console.error(error);
            setFormFeedback(error.message || "删除失败。", "error");
        }
    }

    async function deleteIntegrationById(integrationId) {
        const api = getApi();
        const normalizedId = String(integrationId || "").trim();
        if (!normalizedId) {
            throw new Error("缺少待删除的实例 ID。");
        }
        const record = state.integrations.find((item) => item.id === normalizedId) || null;
        const integrationName = record?.display_name || record?.name || normalizedId;
        if (!window.confirm(`确认删除实例“${integrationName}”吗？此操作会停用相关绑定和路由规则。`)) {
            setFormFeedback("已取消删除。", "info");
            return;
        }
        await api.fetchJson(`/api/integrations/${encodeURIComponent(normalizedId)}`, {
            method: "DELETE",
        });
        await refreshSystemIntegrationsView({ preserveSelection: state.selectedIntegrationId !== normalizedId });
        if (state.selectedIntegrationId === normalizedId) {
            resetIntegrationForm();
        } else {
            renderIntegrationsList();
        }
        setWorkspaceStatus("集成已软删除");
    }

    async function saveBinding(event) {
        event.preventDefault();
        const api = getApi();
        try {
            const integrationId = ensureSelectedIntegration();
            const bindingId = $("#binding-id").value.trim();
            const payload = {
                tenant_id: $("#binding-tenant-id").value.trim(),
                chat_id: $("#binding-chat-id").value.trim(),
                thread_id: $("#binding-thread-id").value.trim(),
                binding_scope: $("#binding-scope").value,
                agent_id: $("#binding-agent-id").value,
            };
            if (!payload.chat_id) {
                throw new Error("绑定的 Chat ID 不能为空。");
            }
            if (!payload.agent_id) {
                throw new Error("请选择目标 Agent。");
            }

            if (bindingId) {
                await api.fetchJson(`/api/bindings/${encodeURIComponent(bindingId)}`, {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        ...payload,
                        refresh_session: $("#binding-refresh-session").checked,
                    }),
                });
            } else {
                await api.fetchJson(`/api/integrations/${encodeURIComponent(integrationId)}/bindings`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
            }
            resetBindingForm();
            await loadSelectedIntegrationResources();
            await refreshSystemIntegrationsView({ preserveSelection: true });
            renderIntegrationsList();
            setWorkspaceStatus("绑定已保存");
        } catch (error) {
            console.error(error);
            setFormFeedback(error.message || "绑定保存失败。", "error");
        }
    }

    async function toggleBinding(bindingId, enabled) {
        const api = getApi();
        await api.fetchJson(`/api/bindings/${encodeURIComponent(bindingId)}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ enabled: !enabled }),
        });
        await loadSelectedIntegrationResources();
        await refreshSystemIntegrationsView({ preserveSelection: true });
        renderIntegrationsList();
    }

    async function deleteBinding(bindingId) {
        const api = getApi();
        await api.fetchJson(`/api/bindings/${encodeURIComponent(bindingId)}`, {
            method: "DELETE",
        });
        await loadSelectedIntegrationResources();
        await refreshSystemIntegrationsView({ preserveSelection: true });
        renderIntegrationsList();
    }

    async function saveRoutingRule(event) {
        event.preventDefault();
        const api = getApi();
        try {
            const integrationId = ensureSelectedIntegration();
            const ruleId = $("#routing-rule-id").value.trim();
            const payload = {
                priority: Number($("#routing-priority").value || 100),
                match_type: $("#routing-match-type").value,
                match_value: $("#routing-match-value").value.trim(),
                agent_id: $("#routing-agent-id").value,
                session_strategy: $("#routing-session-strategy").value,
                enabled: $("#routing-enabled").checked,
                metadata: {},
            };
            if (!payload.match_value) {
                throw new Error("路由匹配值不能为空。");
            }
            if (!payload.agent_id) {
                throw new Error("请选择目标 Agent。");
            }

            if (ruleId) {
                await api.fetchJson(`/api/routing-rules/${encodeURIComponent(ruleId)}`, {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
            } else {
                await api.fetchJson(`/api/integrations/${encodeURIComponent(integrationId)}/routing-rules`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
            }
            resetRoutingForm();
            await loadSelectedIntegrationResources();
            await refreshSystemIntegrationsView({ preserveSelection: true });
            renderIntegrationsList();
            setWorkspaceStatus("路由规则已保存");
        } catch (error) {
            console.error(error);
            setFormFeedback(error.message || "路由规则保存失败。", "error");
        }
    }

    async function toggleRule(ruleId, enabled) {
        const api = getApi();
        await api.fetchJson(`/api/routing-rules/${encodeURIComponent(ruleId)}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ enabled: !enabled }),
        });
        await loadSelectedIntegrationResources();
        await refreshSystemIntegrationsView({ preserveSelection: true });
        renderIntegrationsList();
    }

    async function deleteRule(ruleId) {
        const api = getApi();
        await api.fetchJson(`/api/routing-rules/${encodeURIComponent(ruleId)}`, {
            method: "DELETE",
        });
        await loadSelectedIntegrationResources();
        await refreshSystemIntegrationsView({ preserveSelection: true });
        renderIntegrationsList();
    }

    async function retryDelivery(deliveryId) {
        const api = getApi();
        await api.fetchJson(`/api/outbound-deliveries/${encodeURIComponent(deliveryId)}/retry`, {
            method: "POST",
        });
        await loadSelectedIntegrationResources();
        await refreshSystemIntegrationsView({ preserveSelection: true });
        renderIntegrationsList();
        setWorkspaceStatus("已触发手动重试");
    }

    async function refreshSelectedLogs() {
        await loadSelectedIntegrationResources();
    }

    async function copyWebhookUrl() {
        const input = $("#integration-webhook-url");
        if (!input || !input.value) {
            return;
        }
        const copyValue = input.dataset.copyValue || input.value;
        if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(copyValue);
        } else {
            input.select();
            document.execCommand("copy");
        }
        setFormFeedback("接入地址已复制。", "success");
    }

    function bindEventListeners() {

        const backBtn = $("#back-to-catalog-btn");
        if (backBtn) {
            backBtn.addEventListener("click", () => switchView("catalog"));
        }

        const catalogItems = state.root.querySelectorAll("[data-catalog-item]");
        catalogItems.forEach(item => {
            item.addEventListener("click", () => {
                const kind = item.getAttribute("data-catalog-item");
                openCatalogItem(kind);
            });
        });

        const mgmtTabs = state.root.querySelectorAll(".management-tab");
        mgmtTabs.forEach(tab => {
            tab.addEventListener("click", (e) => {
                e.preventDefault();
                switchTab(tab.getAttribute("data-tab"));
            });
        });

        $("#integration-form")?.addEventListener("submit", saveIntegration);
        $("#binding-form")?.addEventListener("submit", saveBinding);
        $("#routing-form")?.addEventListener("submit", saveRoutingRule);

        $("#integrations-new-btn")?.addEventListener("click", resetIntegrationForm);
        $("#integrations-refresh-btn")?.addEventListener("click", () => refreshSystemIntegrationsView({ preserveSelection: true }).catch(console.error));
        $("#integration-verify-btn")?.addEventListener("click", () => verifySelectedIntegration().catch(console.error));
        $("#integration-enable-btn")?.addEventListener("click", () => updateSelectedIntegrationStatus("active").catch(console.error));
        $("#integration-disable-btn")?.addEventListener("click", () => updateSelectedIntegrationStatus("disabled").catch(console.error));
        $("#integration-delete-btn")?.addEventListener("click", () => deleteSelectedIntegration().catch(console.error));
        $("#integration-copy-webhook-btn")?.addEventListener("click", () => copyWebhookUrl().catch(console.error));
        $("#integration-kind")?.addEventListener("change", () => {
            ensureConfigTemplate(true);
            fillFeishuQuickFields(null);
            fillWeChatQuickFields(null);
            updateIntegrationFormVisibility();
            syncHiddenFeishuConfigFields();
            syncHiddenWeChatConfigFields();
            updateWebhookPreview();
        });
        $("#integration-config-json")?.addEventListener("input", () => updateWebhookPreview());
        $("#integration-feishu-app-id")?.addEventListener("input", () => {
            syncHiddenFeishuConfigFields();
            updateWebhookPreview();
        });
        $("#integration-feishu-default-chat-id")?.addEventListener("input", () => {
            syncHiddenFeishuConfigFields();
        });
        $("#integration-feishu-default-thread-id")?.addEventListener("input", () => {
            syncHiddenFeishuConfigFields();
        });
        $("#integration-feishu-app-secret")?.addEventListener("input", () => {
            const target = $("#integration-feishu-app-secret-status");
            if (target) {
                target.textContent = $("#integration-feishu-app-secret")?.value.trim()
                    ? "保存后会更新 App Secret。"
                    : "新建时必填；编辑现有集成时留空表示保持不变。";
            }
        });
        $("#integration-default-agent-id")?.addEventListener("change", () => {
            syncHiddenFeishuConfigFields();
        });
        $("#integration-wechat-default-agent-id")?.addEventListener("change", () => {
            syncHiddenWeChatConfigFields();
        });
        $("#integration-wechat-setup-btn")?.addEventListener("click", () => startWeChatSetup().catch(console.error));

        $("#integration-add-credential-btn")?.addEventListener("click", () => {
            const container = $("#integration-credential-rows");
            if (container) {
                container.insertAdjacentHTML("beforeend", credentialRowTemplate({}));
            }
        });

        $("#binding-reset-btn")?.addEventListener("click", resetBindingForm);
        $("#routing-reset-btn")?.addEventListener("click", resetRoutingForm);
        $("#events-refresh-btn")?.addEventListener("click", () => refreshSelectedLogs().catch(console.error));
        $("#deliveries-refresh-btn")?.addEventListener("click", () => refreshSelectedLogs().catch(console.error));
        $("#events-status-filter")?.addEventListener("change", () => refreshSelectedLogs().catch(console.error));
        $("#deliveries-status-filter")?.addEventListener("change", () => refreshSelectedLogs().catch(console.error));

        state.root.addEventListener("click", (event) => {
            const inlineDeleteButton = event.target.closest('[data-action="delete-integration-inline"]');
            if (inlineDeleteButton) {
                event.preventDefault();
                event.stopPropagation();
                deleteIntegrationById(
                    inlineDeleteButton.getAttribute("data-integration-id"),
                ).catch(console.error);
                return;
            }

            const integrationButton = event.target.closest("[data-integration-select]");
            if (integrationButton) {
                selectIntegration(integrationButton.getAttribute("data-integration-select")).catch(console.error);
                return;
            }

            const removeCredentialButton = event.target.closest('[data-action="remove-credential"]');
            if (removeCredentialButton) {
                removeCredentialButton.closest("[data-credential-row]")?.remove();
                if (!state.root.querySelector("[data-credential-row]")) {
                    renderCredentialRows([{}]);
                }
                return;
            }

            const editBindingButton = event.target.closest('[data-action="edit-binding"]');
            if (editBindingButton) {
                const binding = state.bindings.find((item) => item.id === editBindingButton.getAttribute("data-binding-id"));
                if (binding) {
                    fillBindingForm(binding);
                }
                return;
            }

            const toggleBindingButton = event.target.closest('[data-action="toggle-binding"]');
            if (toggleBindingButton) {
                toggleBinding(
                    toggleBindingButton.getAttribute("data-binding-id"),
                    toggleBindingButton.getAttribute("data-enabled") === "true",
                ).catch(console.error);
                return;
            }

            const deleteBindingButton = event.target.closest('[data-action="delete-binding"]');
            if (deleteBindingButton) {
                deleteBinding(deleteBindingButton.getAttribute("data-binding-id")).catch(console.error);
                return;
            }

            const editRuleButton = event.target.closest('[data-action="edit-rule"]');
            if (editRuleButton) {
                const rule = state.routingRules.find((item) => item.id === editRuleButton.getAttribute("data-rule-id"));
                if (rule) {
                    fillRoutingForm(rule);
                }
                return;
            }

            const toggleRuleButton = event.target.closest('[data-action="toggle-rule"]');
            if (toggleRuleButton) {
                toggleRule(
                    toggleRuleButton.getAttribute("data-rule-id"),
                    toggleRuleButton.getAttribute("data-enabled") === "true",
                ).catch(console.error);
                return;
            }

            const deleteRuleButton = event.target.closest('[data-action="delete-rule"]');
            if (deleteRuleButton) {
                deleteRule(deleteRuleButton.getAttribute("data-rule-id")).catch(console.error);
                return;
            }

            const retryDeliveryButton = event.target.closest('[data-action="retry-delivery"]');
            if (retryDeliveryButton) {
                retryDelivery(retryDeliveryButton.getAttribute("data-delivery-id")).catch(console.error);
            }
        });
    }

    async function refreshSystemIntegrationsView({ preserveSelection = true } = {}) {
        await refreshAgentsAndIntegrations({ preserveSelection });
        if (state.selectedIntegrationId) {
            const selected = getSelectedIntegration();
            if (selected) {
                fillIntegrationForm(selected);
                await loadSelectedIntegrationResources();
                if (isWeChatKind(selected)) {
                    renderWeChatSetupStatus(selected.setup_status || null);
                    if (selected.setup_status && ["queued", "running", "waiting_scan"].includes(selected.setup_status.state)) {
                        await pollWeChatSetupStatus();
                    }
                }
                return;
            }
        }
        stopWeChatSetupPolling();
        fillIntegrationForm(null);
        renderBindingsList();
        renderRoutingRulesList();
        renderEventsList();
        renderDeliveriesList();
    }

    function initSystemIntegrationsView(root) {
        if (!root) {
            return;
        }
        state.root = root;
        if (!root.dataset.integrationsBound) {
            bindEventListeners();
            root.dataset.integrationsBound = "true";
        }
        ensureConfigTemplate(true);
        updateWeChatCatalogCard();
        updateWebhookPreview();
        switchView("catalog");
        refreshSystemIntegrationsView({ preserveSelection: true }).catch((error) => {
            console.error("Failed to initialize integrations view:", error);
            setFormFeedback(error.message || "Integrations 页面初始化失败。", "error");
        });
    }

    window.initSystemIntegrationsView = initSystemIntegrationsView;
    window.refreshSystemIntegrationsView = refreshSystemIntegrationsView;
})();


(() => {
    const state = {
        root: null,
        tasks: [],
        integrations: [],
        executions: [],
        selectedTaskId: "",
    };

    function $(selector) {
        return state.root ? state.root.querySelector(selector) : null;
    }

    function getApi() {
        return window.ClaviAgentWorkspaceApi || { fetchJson: async () => ({}) };
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
        if (!value) return "未触发";
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return String(value);
        return date.toLocaleString("zh-CN", {
            year: "numeric",
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
            hour12: false,
        });
    }

    function statusBadge(status) {
        const normalized = String(status || "").toLowerCase();
        if (["completed", "delivered", "active", "enabled"].includes(normalized)) {
            return {
                text: normalized || "completed",
                className: "border-emerald-400/20 bg-emerald-400/10 text-emerald-300",
            };
        }
        if (["queued", "running", "waiting_approval", "pending"].includes(normalized)) {
            return {
                text: normalized || "pending",
                className: "border-primary/20 bg-primary/10 text-primary",
            };
        }
        if (["failed", "cancelled", "timed_out", "dispatch_failed", "error"].includes(normalized)) {
            return {
                text: normalized || "failed",
                className: "border-error/20 bg-error/10 text-error",
            };
        }
        return {
            text: normalized || "unknown",
            className: "border-white/10 bg-white/5 text-on-surface-variant",
        };
    }

    function switchTab(tabName) {
        const tabs = state.root.querySelectorAll(".task-tab");
        const contents = state.root.querySelectorAll(".tab-content");
        tabs.forEach((tab) => {
            if (tab.dataset.tab === tabName) {
                tab.classList.add("active", "text-primary", "border-primary");
                tab.classList.remove("text-on-surface-variant", "border-transparent");
            } else {
                tab.classList.remove("active", "text-primary", "border-primary");
                tab.classList.add("text-on-surface-variant", "border-transparent");
            }
        });
        contents.forEach((content) => {
            if (content.id === `tab-${tabName}`) {
                content.classList.remove("hidden");
                content.classList.add(tabName === "config" ? "block" : "flex");
            } else {
                content.classList.add("hidden");
                content.classList.remove("block", "flex");
            }
        });
        if (tabName === "logs" && state.selectedTaskId) {
            loadExecutions(state.selectedTaskId).catch((error) => {
                console.error("Failed to load scheduled task executions:", error);
                renderExecutions();
            });
        }
    }

    function showFormFeedback(message, mode = "info") {
        const feedback = $("#task-form-feedback");
        if (!feedback) return;
        feedback.classList.remove("hidden");
        feedback.textContent = message;
        if (mode === "success") {
            feedback.className = "mt-4 rounded-xl border border-emerald-400/20 bg-emerald-400/10 px-4 py-2.5 text-xs text-emerald-300 w-full";
        } else if (mode === "error") {
            feedback.className = "mt-4 rounded-xl border border-error/20 bg-error/10 px-4 py-2.5 text-xs text-error w-full";
        } else {
            feedback.className = "mt-4 rounded-xl border border-outline-variant/10 bg-black/20 px-4 py-2.5 text-xs text-on-surface-variant w-full";
        }
        window.clearTimeout(showFormFeedback._timer);
        showFormFeedback._timer = window.setTimeout(() => {
            feedback.classList.add("hidden");
        }, 5000);
    }

    function setElementVisible(selector, visible) {
        const element = $(selector);
        if (!element) return;
        element.classList.toggle("hidden", !visible);
    }

    function getSelectedTask() {
        return state.tasks.find((item) => item.id === state.selectedTaskId) || null;
    }

    function getIntegrationById(integrationId) {
        return state.integrations.find((item) => item.id === integrationId) || null;
    }

    function integrationLabel(integration) {
        if (!integration) {
            return "";
        }
        return integration.display_name || integration.name || integration.id;
    }

    function resolveIntegrationDefaultAgentId(integration) {
        const config = integration?.config || {};
        return String(
            config.default_agent_id
            || config.default_agent_template_id
            || ""
        ).trim();
    }

    function resolveDeliveryTarget(integration, task = null) {
        const config = integration?.config || {};
        return {
            chatId: String(
                task?.target_chat_id
                || task?.resolved_target_chat_id
                || config.default_chat_id
                || config.default_target_id
                || config.target_id
                || config.receive_id
                || ""
            ).trim(),
            threadId: String(
                task?.target_thread_id
                || task?.resolved_target_thread_id
                || config.default_thread_id
                || config.thread_id
                || ""
            ).trim(),
        };
    }

    function getIntegrationIssues(integration, task = null, { allowStoredAgentFallback = false } = {}) {
        if (!integration) {
            return ["发送渠道不存在或已被删除"];
        }
        const issues = [];
        const resolvedAgentId = String(
            task?.agent_id && allowStoredAgentFallback
                ? task.agent_id
                : resolveIntegrationDefaultAgentId(integration)
        ).trim();
        const target = resolveDeliveryTarget(integration, task);

        if (!resolvedAgentId) {
            issues.push("该渠道未绑定默认 Agent");
        }
        if (!target.chatId) {
            issues.push("该渠道未配置默认发送目标（首次收到消息后可自动回填）");
        }
        if (String(integration.status || "").trim().toLowerCase() !== "active") {
            issues.push(`渠道当前状态为 ${integration.status || "unknown"}`);
        }
        return issues;
    }

    function getBlockingIntegrationIssues(integration, task = null, { allowStoredAgentFallback = false } = {}) {
        return getIntegrationIssues(integration, task, { allowStoredAgentFallback })
            .filter((issue) => !issue.startsWith("渠道当前状态为 "));
    }

    function renderIntegrationSummary(task = null) {
        const panel = $("#task-integration-summary");
        if (!panel) return;

        const integrationId = String($("#task-integration-id")?.value || task?.integration_id || "").trim();
        if (!integrationId) {
            panel.className = "rounded-[18px] border border-white/5 bg-black/20 px-4 py-4 text-xs leading-relaxed text-on-surface-variant";
            panel.innerHTML = "选择发送渠道后，这里会显示当前渠道绑定的 Agent 和默认发送目标。";
            return;
        }

        const integration = getIntegrationById(integrationId);
        const issues = getIntegrationIssues(integration, task, {
            allowStoredAgentFallback: integrationId === String(task?.integration_id || "").trim(),
        });
        if (!integration) {
            panel.className = "rounded-[18px] border border-error/20 bg-error/10 px-4 py-4 text-xs leading-relaxed text-error";
            panel.textContent = "所选发送渠道不存在或已被删除。";
            return;
        }

        const badge = statusBadge(integration.status || "unknown");
        const resolvedAgentId = String(
            task?.agent_id && integrationId === String(task?.integration_id || "").trim()
                ? task.agent_id
                : resolveIntegrationDefaultAgentId(integration)
        ).trim();
        const target = resolveDeliveryTarget(integration, task);
        const toneClass = issues.length
            ? "rounded-[18px] border border-error/20 bg-error/10 px-4 py-4 text-xs leading-relaxed text-error"
            : "rounded-[18px] border border-emerald-400/20 bg-emerald-400/10 px-4 py-4 text-xs leading-relaxed text-emerald-200";

        panel.className = toneClass;
        panel.innerHTML = `
            <div class="flex flex-wrap items-center gap-2">
                <span class="text-[11px] font-bold uppercase tracking-[0.18em]">Channel Binding</span>
                <span class="inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-bold ${badge.className}">
                    ${escapeHtml(badge.text)}
                </span>
                <span class="text-[11px] text-white">${escapeHtml(integrationLabel(integration))}</span>
            </div>
            <div class="mt-3 grid gap-2 md:grid-cols-2">
                <div class="rounded-xl border border-white/10 bg-black/20 px-3 py-2">
                    <div class="text-[10px] uppercase tracking-[0.16em] text-on-surface-variant">Bound Agent</div>
                    <div class="mt-1 break-all text-sm font-semibold text-white">${escapeHtml(resolvedAgentId || "未配置")}</div>
                </div>
                <div class="rounded-xl border border-white/10 bg-black/20 px-3 py-2">
                    <div class="text-[10px] uppercase tracking-[0.16em] text-on-surface-variant">Default Chat</div>
                    <div class="mt-1 break-all text-sm font-semibold text-white">${escapeHtml(target.chatId || "未配置")}</div>
                </div>
                ${target.threadId ? `
                    <div class="rounded-xl border border-white/10 bg-black/20 px-3 py-2 md:col-span-2">
                        <div class="text-[10px] uppercase tracking-[0.16em] text-on-surface-variant">Default Thread</div>
                        <div class="mt-1 break-all text-sm font-semibold text-white">${escapeHtml(target.threadId)}</div>
                    </div>
                ` : ""}
            </div>
            <div class="mt-3 ${issues.length ? "text-error" : "text-emerald-200"}">
                ${issues.length
                    ? `保存或触发前请先修正：${escapeHtml(issues.join("；"))}`
                    : "当前渠道配置完整，任务会直接继承该渠道绑定的 Agent 和默认发送目标。"}
            </div>
        `;
    }

    async function fetchReferenceData() {
        const api = getApi();
        const [integrations, tasks] = await Promise.all([
            api.fetchJson("/api/integrations"),
            api.fetchJson("/api/scheduled-tasks"),
        ]);
        state.integrations = integrations || [];
        state.tasks = tasks || [];
    }

    async function loadExecutions(taskId) {
        const api = getApi();
        if (!taskId) {
            state.executions = [];
            return;
        }
        state.executions = await api.fetchJson(
            `/api/scheduled-tasks/${encodeURIComponent(taskId)}/executions?limit=20`
        );
    }

    function renderSelectOptions(selectId, items, noneLabel = "请选择发送渠道") {
        const element = $(selectId);
        if (!element) return;
        const currentValue = element.value;
        const optionsHtml = items.map((item) => {
            const label = integrationLabel(item);
            const status = String(item.status || "").trim().toLowerCase();
            const suffix = status && status !== "active" ? ` (${status})` : "";
            return `<option value="${escapeHtml(item.id)}">${escapeHtml(label + suffix)}</option>`;
        }).join("");
        element.innerHTML = `<option value="">${escapeHtml(noneLabel)}</option>${optionsHtml}`;
        if (currentValue) {
            element.value = currentValue;
        }
    }

    function fillForm(task = null) {
        const isEditing = Boolean(task);
        $("#task-id").value = task?.id || "";
        $("#task-name").value = task?.name || "";
        const cronValue = task?.cron_expression || "";
        const presetEl = $("#task-cron-preset");
        const cronInput = $("#task-cron");
        if (cronInput) cronInput.value = cronValue;

        if (presetEl) {
            let found = false;
            for (const opt of presetEl.options) {
                if (opt.value === cronValue && cronValue !== "custom") {
                    found = true;
                    break;
                }
            }
            presetEl.value = found ? cronValue : "custom";
        }
        $("#task-prompt").value = task?.prompt || "";
        $("#task-integration-id").value = task?.integration_id || "";
        $("#task-panel-title").textContent = isEditing ? task.name : "新建任务";
        renderIntegrationSummary(task);

        const statusElement = $("#task-current-status");
        if (!statusElement) return;
        if (!isEditing) {
            statusElement.textContent = "未保存";
            statusElement.className = "inline-flex items-center rounded-full border border-outline-variant/20 bg-black/20 px-3 py-1 text-[11px] font-bold text-on-surface-variant tracking-wider";
        } else if (task.enabled) {
            statusElement.textContent = "已启用";
            statusElement.className = "inline-flex items-center rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-[11px] font-bold text-emerald-300 tracking-wider";
        } else {
            statusElement.textContent = "已暂停";
            statusElement.className = "inline-flex items-center rounded-full border border-secondary/20 bg-secondary/10 px-3 py-1 text-[11px] font-bold text-secondary tracking-wider";
        }

        setElementVisible("#task-run-btn", isEditing);
        setElementVisible("#task-enable-btn", isEditing && !task?.enabled);
        setElementVisible("#task-disable-btn", isEditing && task?.enabled);
        setElementVisible("#task-delete-btn", isEditing);
    }

    function renderTasksList() {
        const container = $("#tasks-list");
        if (!container) return;
        if (!state.tasks.length) {
            container.innerHTML = `
                <div class="rounded-2xl border border-dashed border-outline-variant/20 bg-black/20 px-4 py-8 text-center text-[11px] leading-relaxed text-on-surface-variant">
                    暂无定时任务。<br>点击上方按钮创建第一条任务。
                </div>
            `;
            return;
        }

        container.innerHTML = state.tasks.map((task) => {
            const isActive = task.id === state.selectedTaskId;
            const styleActive = isActive
                ? "border-primary/40 bg-primary/10 shadow-[0_0_18px_rgba(0,240,255,0.08)]"
                : "border-white/5 bg-black/20 hover:border-primary/20 hover:bg-black/40";
            const badge = statusBadge(task.enabled ? "running" : "disabled");
            const latestStatus = task.last_execution?.status || "";
            const nextRun = task.next_run_at ? formatTime(task.next_run_at) : "未计划";
            const channelName = task.integration_display_name || task.integration_id || "未绑定渠道";
            const targetLabel = task.resolved_target_chat_id
                ? ` -> ${task.resolved_target_chat_id}`
                : "";
            return `
                <button type="button" data-task-id="${escapeHtml(task.id)}"
                    class="w-full rounded-[16px] border px-4 py-4 text-left transition-all relative overflow-hidden ${styleActive}">
                    ${isActive ? '<div class="absolute left-0 top-0 bottom-0 w-1 bg-primary"></div>' : ""}
                    <div class="flex flex-col gap-2">
                        <div class="flex items-start justify-between gap-2">
                            <span class="text-sm font-bold text-white truncate break-all">${escapeHtml(task.name)}</span>
                            <span class="shrink-0 inline-flex items-center rounded-full border px-2 py-0.5 text-[9px] font-bold ${badge.className}">
                                ${task.enabled ? "启用中" : "已暂停"}
                            </span>
                        </div>
                        <div class="text-[10px] text-on-surface-variant font-mono truncate">
                            <span class="material-symbols-outlined text-[10px] align-middle">schedule</span>
                            ${escapeHtml(task.cron_expression)}
                        </div>
                        <div class="text-[10px] text-on-surface-variant truncate">
                            下次执行：${escapeHtml(nextRun)}
                        </div>
                        <div class="text-[10px] text-on-surface-variant truncate">
                            渠道：${escapeHtml(channelName + targetLabel)}
                        </div>
                        ${latestStatus ? `<div class="text-[10px] text-on-surface-variant truncate">最近状态：${escapeHtml(latestStatus)}</div>` : ""}
                    </div>
                </button>
            `;
        }).join("");
    }

    function renderExecutions() {
        const container = $("#task-logs-list");
        if (!container) return;
        if (!state.selectedTaskId) {
            container.innerHTML = `
                <div class="absolute inset-0 flex flex-col items-center justify-center text-sm text-on-surface-variant opacity-70">
                    <span class="material-symbols-outlined text-4xl mb-4">task_alt</span>
                    <p>请选择左侧定时任务后查看执行历史。</p>
                </div>
            `;
            return;
        }
        if (!state.executions.length) {
            container.innerHTML = `
                <div class="absolute inset-0 flex flex-col items-center justify-center text-sm text-on-surface-variant opacity-70">
                    <span class="material-symbols-outlined text-4xl mb-4">hourglass_empty</span>
                    <p>当前任务还没有执行记录。</p>
                </div>
            `;
            return;
        }

        container.innerHTML = state.executions.map((execution) => {
            const runBadge = statusBadge(execution.status);
            const deliveryBadge = execution.delivery_status
                ? statusBadge(execution.delivery_status)
                : null;
            return `
                <div class="rounded-2xl border border-white/5 bg-black/20 px-4 py-4">
                    <div class="flex items-start justify-between gap-3">
                        <div class="space-y-2 min-w-0">
                            <div class="flex items-center gap-2 flex-wrap">
                                <span class="inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-bold ${runBadge.className}">
                                    ${escapeHtml(runBadge.text)}
                                </span>
                                <span class="text-[10px] uppercase tracking-[0.16em] text-on-surface-variant">
                                    ${escapeHtml(execution.trigger_kind)}
                                </span>
                                ${deliveryBadge ? `
                                    <span class="inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-bold ${deliveryBadge.className}">
                                        delivery:${escapeHtml(deliveryBadge.text)}
                                    </span>
                                ` : ""}
                            </div>
                            <div class="text-xs font-semibold text-white break-all">
                                Run ID: ${escapeHtml(execution.run_id || "未创建")}
                            </div>
                            <div class="text-[11px] text-on-surface-variant leading-relaxed">
                                触发时间：${escapeHtml(formatTime(execution.created_at))}<br>
                                开始时间：${escapeHtml(formatTime(execution.started_at))}<br>
                                结束时间：${escapeHtml(formatTime(execution.finished_at))}
                            </div>
                            ${execution.run_error_summary ? `
                                <div class="rounded-xl border border-error/20 bg-error/10 px-3 py-2 text-[11px] text-error leading-relaxed">
                                    ${escapeHtml(execution.run_error_summary)}
                                </div>
                            ` : ""}
                        </div>
                        <div class="shrink-0 text-[10px] text-on-surface-variant text-right">
                            <div>投递数：${escapeHtml(execution.delivery_count)}</div>
                            ${execution.scheduled_for ? `<div>计划时间：${escapeHtml(formatTime(execution.scheduled_for))}</div>` : ""}
                        </div>
                    </div>
                </div>
            `;
        }).join("");
    }

    async function selectTask(taskId) {
        state.selectedTaskId = taskId;
        const task = state.tasks.find((item) => item.id === taskId);
        if (task) {
            fillForm(task);
            await loadExecutions(taskId);
            renderExecutions();
            switchTab("config");
        } else {
            resetForm();
        }
        renderTasksList();
    }

    function resetForm() {
        state.selectedTaskId = "";
        state.executions = [];
        fillForm(null);
        renderTasksList();
        renderExecutions();
        switchTab("config");
    }

    function collectPayloadFromForm() {
        return {
            name: $("#task-name").value.trim(),
            cron_expression: $("#task-cron").value.trim(),
            integration_id: $("#task-integration-id").value || null,
            prompt: $("#task-prompt").value.trim(),
        };
    }

    function validateFormPayload(payload, existingTask = null) {
        if (!payload.name || !payload.cron_expression || !payload.prompt) {
            return "请完整填写必填字段。";
        }
        if (!payload.integration_id) {
            if (existingTask?.agent_id) {
                return "";
            }
            return "请选择发送渠道。";
        }
        const integration = getIntegrationById(payload.integration_id);
        const issues = getBlockingIntegrationIssues(
            integration,
            existingTask && payload.integration_id === existingTask.integration_id ? existingTask : null,
            {
                allowStoredAgentFallback: payload.integration_id === existingTask?.integration_id,
            }
        );
        return issues.length ? issues[0] : "";
    }

    async function handleFormSubmit(event) {
        event.preventDefault();
        const api = getApi();
        const id = $("#task-id").value;
        const existingTask = id ? state.tasks.find((item) => item.id === id) || null : null;
        const payload = collectPayloadFromForm();
        const validationError = validateFormPayload(payload, existingTask);
        if (validationError) {
            showFormFeedback(validationError, "error");
            renderIntegrationSummary(existingTask);
            return;
        }
        try {
            const savedTask = id
                ? await api.fetchJson(`/api/scheduled-tasks/${encodeURIComponent(id)}`, {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                })
                : await api.fetchJson("/api/scheduled-tasks", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
            await fetchReferenceData();
            state.selectedTaskId = savedTask.id;
            fillForm(savedTask);
            renderTasksList();
            showFormFeedback(id ? "任务更新成功。" : "任务创建成功。", "success");
        } catch (error) {
            showFormFeedback(error.message || "保存任务失败。", "error");
        }
    }

    async function toggleTaskEnabled(enabled) {
        if (!state.selectedTaskId) return;
        const api = getApi();
        const endpoint = enabled ? "enable" : "disable";
        try {
            const task = await api.fetchJson(
                `/api/scheduled-tasks/${encodeURIComponent(state.selectedTaskId)}/${endpoint}`,
                { method: "POST" }
            );
            await fetchReferenceData();
            fillForm(task);
            renderTasksList();
            showFormFeedback(enabled ? "任务已启用。" : "任务已暂停。", "success");
        } catch (error) {
            showFormFeedback(error.message || "更新任务状态失败。", "error");
        }
    }

    async function deleteSelectedTask() {
        if (!state.selectedTaskId) return;
        if (!window.confirm("确定要删除这条定时任务吗？")) return;
        const api = getApi();
        try {
            await api.fetchJson(`/api/scheduled-tasks/${encodeURIComponent(state.selectedTaskId)}`, {
                method: "DELETE",
            });
            await fetchReferenceData();
            resetForm();
            showFormFeedback("任务已删除。", "success");
        } catch (error) {
            showFormFeedback(error.message || "删除任务失败。", "error");
        }
    }

    async function runSelectedTaskNow() {
        if (!state.selectedTaskId) return;
        const api = getApi();
        try {
            const execution = await api.fetchJson(
                `/api/scheduled-tasks/${encodeURIComponent(state.selectedTaskId)}/run`,
                { method: "POST" }
            );
            await fetchReferenceData();
            await loadExecutions(state.selectedTaskId);
            renderTasksList();
            renderExecutions();
            showFormFeedback(
                execution.run_id
                    ? "已下发立即执行任务，可在下方查看执行历史。"
                    : "任务执行已创建，但调度失败，请检查错误信息。",
                execution.run_id ? "success" : "error"
            );
            switchTab("logs");
        } catch (error) {
            showFormFeedback(error.message || "立即执行失败。", "error");
        }
    }

    function bindEvents() {
        const root = state.root;
        root.querySelectorAll(".task-tab").forEach((tab) => {
            tab.addEventListener("click", () => switchTab(tab.dataset.tab));
        });

        $("#tasks-new-btn")?.addEventListener("click", resetForm);
        $("#task-cron-preset")?.addEventListener("change", (e) => {
            const val = e.target.value;
            const cronInput = $("#task-cron");
            if (!cronInput) return;
            if (val !== "custom") {
                cronInput.value = val;
            } else {
                cronInput.focus();
            }
        });

        $("#task-cron")?.addEventListener("input", (e) => {
            const val = e.target.value.trim();
            const presetEl = $("#task-cron-preset");
            if (!presetEl) return;
            let found = false;
            for (const opt of presetEl.options) {
                if (opt.value === val && val !== "custom") {
                    found = true;
                    presetEl.value = val;
                    break;
                }
            }
            if (!found) {
                presetEl.value = "custom";
            }
        });
        $("#task-integration-id")?.addEventListener("change", () => {
            renderIntegrationSummary(getSelectedTask());
        });
        $("#task-form")?.addEventListener("submit", (event) => {
            handleFormSubmit(event).catch((error) => {
                showFormFeedback(error.message || "保存任务失败。", "error");
            });
        });
        $("#task-enable-btn")?.addEventListener("click", () => {
            toggleTaskEnabled(true);
        });
        $("#task-disable-btn")?.addEventListener("click", () => {
            toggleTaskEnabled(false);
        });
        $("#task-delete-btn")?.addEventListener("click", () => {
            deleteSelectedTask();
        });
        $("#task-run-btn")?.addEventListener("click", () => {
            runSelectedTaskNow();
        });
        $("#task-logs-refresh-btn")?.addEventListener("click", () => {
            if (!state.selectedTaskId) return;
            loadExecutions(state.selectedTaskId)
                .then(() => renderExecutions())
                .catch((error) => showFormFeedback(error.message || "刷新执行历史失败。", "error"));
        });

        $("#tasks-list")?.addEventListener("click", (event) => {
            const button = event.target.closest("button[data-task-id]");
            if (!button) return;
            selectTask(button.dataset.taskId).catch((error) => {
                showFormFeedback(error.message || "加载任务失败。", "error");
            });
        });
    }

    window.initScheduledTasksView = async function(rootElement) {
        state.root = rootElement;
        await window.refreshScheduledTasksView();
        bindEvents();
    };

    window.refreshScheduledTasksView = async function() {
        await fetchReferenceData();
        renderSelectOptions("#task-integration-id", state.integrations, "请选择发送渠道");
        renderTasksList();
        if (state.selectedTaskId) {
            const task = state.tasks.find((item) => item.id === state.selectedTaskId);
            if (task) {
                fillForm(task);
                await loadExecutions(task.id).catch(() => {
                    state.executions = [];
                });
                renderExecutions();
                return;
            }
        }
        resetForm();
    };
})();


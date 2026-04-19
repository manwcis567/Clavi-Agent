(() => {
    function escapeHtml(value) {
        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function truncate(value, maxLength = 72) {
        const text = String(value ?? "").trim();
        if (text.length <= maxLength) {
            return text;
        }
        return `${text.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`;
    }

    function getStatusMeta(status) {
        const normalized = String(status || "").trim();
        const mappings = {
            queued: {
                label: "排队中",
                tone: "border-slate-300/20 bg-slate-300/10 text-slate-200",
                stroke: "rgba(148, 163, 184, 0.55)",
            },
            running: {
                label: "进行中",
                tone: "border-cyan-300/25 bg-cyan-300/10 text-cyan-200",
                stroke: "rgba(103, 232, 249, 0.85)",
            },
            waiting_approval: {
                label: "待审批",
                tone: "border-amber-300/25 bg-amber-300/10 text-amber-200",
                stroke: "rgba(252, 211, 77, 0.8)",
            },
            interrupted: {
                label: "待继续",
                tone: "border-fuchsia-300/25 bg-fuchsia-300/10 text-fuchsia-200",
                stroke: "rgba(232, 121, 249, 0.8)",
            },
            timed_out: {
                label: "超时",
                tone: "border-orange-300/25 bg-orange-300/10 text-orange-200",
                stroke: "rgba(253, 186, 116, 0.8)",
            },
            completed: {
                label: "已完成",
                tone: "border-emerald-300/25 bg-emerald-300/10 text-emerald-200",
                stroke: "rgba(110, 231, 183, 0.85)",
            },
            failed: {
                label: "失败",
                tone: "border-rose-300/25 bg-rose-300/10 text-rose-200",
                stroke: "rgba(251, 113, 133, 0.85)",
            },
            cancelled: {
                label: "已取消",
                tone: "border-zinc-300/20 bg-zinc-300/10 text-zinc-200",
                stroke: "rgba(212, 212, 216, 0.6)",
            },
        };
        return mappings[normalized] || mappings.completed;
    }

    function annotateLeafCounts(node) {
        if (!node || typeof node !== "object") {
            return null;
        }
        const children = Array.isArray(node.children)
            ? node.children.map((child) => annotateLeafCounts(child)).filter(Boolean)
            : [];
        const leafCount = children.length
            ? children.reduce((sum, child) => sum + child.__leafCount, 0)
            : 1;
        return {
            ...node,
            children,
            __leafCount: leafCount,
        };
    }

    function buildDagLayout(rootNode) {
        const annotatedRoot = annotateLeafCounts(rootNode);
        if (!annotatedRoot) {
            return null;
        }

        const config = {
            paddingX: 20,
            paddingY: 18,
            nodeWidth: 188,
            nodeHeight: 108,
            gapX: 32,
            gapY: 36,
        };

        const nodes = [];
        const edges = [];
        let maxDepth = 0;

        function placeNode(node, depth, leafStart) {
            const childCount = Array.isArray(node.children) ? node.children.length : 0;
            const centerLeaf = leafStart + (node.__leafCount - 1) / 2;
            const centerX =
                config.paddingX +
                centerLeaf * (config.nodeWidth + config.gapX) +
                config.nodeWidth / 2;
            const left = Math.round(centerX - config.nodeWidth / 2);
            const top = config.paddingY + depth * (config.nodeHeight + config.gapY);

            const layoutNode = {
                ...node,
                childCount,
                depth,
                left,
                top,
                centerX: Math.round(centerX),
                centerY: Math.round(top + config.nodeHeight / 2),
            };

            nodes.push(layoutNode);
            maxDepth = Math.max(maxDepth, depth);

            let currentLeafStart = leafStart;
            for (const child of node.children || []) {
                const childLayoutNode = placeNode(child, depth + 1, currentLeafStart);
                edges.push({
                    fromX: layoutNode.centerX,
                    fromY: layoutNode.top + config.nodeHeight,
                    toX: childLayoutNode.centerX,
                    toY: childLayoutNode.top,
                    stroke: getStatusMeta(childLayoutNode.status).stroke,
                });
                currentLeafStart += child.__leafCount;
            }

            return layoutNode;
        }

        placeNode(annotatedRoot, 0, 0);

        return {
            nodes,
            edges,
            width:
                config.paddingX * 2 +
                annotatedRoot.__leafCount * config.nodeWidth +
                Math.max(0, annotatedRoot.__leafCount - 1) * config.gapX,
            height:
                config.paddingY * 2 +
                (maxDepth + 1) * config.nodeHeight +
                Math.max(0, maxDepth) * config.gapY,
            nodeWidth: config.nodeWidth,
            nodeHeight: config.nodeHeight,
            maxDepth,
        };
    }

    function renderEdge(edge, index) {
        const midY = Math.round((edge.fromY + edge.toY) / 2);
        const path = [
            `M ${edge.fromX} ${edge.fromY}`,
            `C ${edge.fromX} ${midY}, ${edge.toX} ${midY}, ${edge.toX} ${edge.toY}`,
        ].join(" ");

        return `
            <g data-edge-index="${index}">
                <path
                    d="${path}"
                    fill="none"
                    stroke="${edge.stroke}"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-dasharray="4 6"
                    opacity="0.8"
                ></path>
                <circle cx="${edge.fromX}" cy="${edge.fromY}" r="3" fill="${edge.stroke}" opacity="0.9"></circle>
                <circle cx="${edge.toX}" cy="${edge.toY}" r="3" fill="${edge.stroke}" opacity="0.9"></circle>
            </g>
        `;
    }

    function renderNode(node, selectedRunId) {
        const meta = getStatusMeta(node.status);
        const isSelected = node.id === selectedRunId;
        const label = truncate(node.goal || node.agent_name || node.id || "run", 60);
        const latestEvent = node.latest_trace_event?.event_type
            ? truncate(node.latest_trace_event.event_type.replace(/_/g, " "), 24)
            : "无最新事件";

        return `
            <button
                type="button"
                data-run-action="select"
                data-run-id="${escapeHtml(node.id)}"
                title="${escapeHtml(node.goal || node.id || "运行节点")}"
                class="absolute rounded-2xl border px-3 py-3 text-left transition-all ${
                    isSelected
                        ? "border-primary/60 bg-surface-container-highest shadow-[0_0_0_1px_rgba(0,238,252,0.25),0_14px_34px_rgba(0,238,252,0.18)]"
                        : "border-outline-variant/10 bg-surface/90 hover:border-primary/35 hover:bg-surface-container-high"
                }"
                style="left:${node.left}px;top:${node.top}px;width:188px;height:108px"
            >
                <div class="flex items-start justify-between gap-2">
                    <div class="min-w-0">
                        <div class="text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">${escapeHtml(
                            truncate(node.agent_name || "main", 18)
                        )}</div>
                        <div
                            class="mt-2 overflow-hidden text-sm font-semibold leading-snug text-on-surface"
                            style="display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical"
                        >${escapeHtml(label)}</div>
                    </div>
                    <span class="inline-flex shrink-0 items-center rounded-full border px-2 py-1 text-[10px] font-mono uppercase tracking-[0.16em] ${meta.tone}">${escapeHtml(
                        meta.label
                    )}</span>
                </div>
                <div class="mt-3 flex flex-wrap gap-1 text-[10px] font-mono uppercase tracking-[0.16em] text-on-surface-variant">
                    <span>${escapeHtml(String(node.id || "").slice(0, 8))}</span>
                    <span>/</span>
                    <span>${escapeHtml(`${node.childCount} 子任务`)}</span>
                    <span>/</span>
                    <span>${escapeHtml(`${node.artifact_count || 0} 产出`)}</span>
                </div>
                <div class="mt-2 text-[10px] text-on-surface-variant">${escapeHtml(latestEvent)}</div>
            </button>
        `;
    }

    function countNodes(node) {
        if (!node || typeof node !== "object") {
            return 0;
        }
        const children = Array.isArray(node.children) ? node.children : [];
        return 1 + children.reduce((sum, child) => sum + countNodes(child), 0);
    }

    function renderSection({ tree, selectedRunId = "" } = {}) {
        if (!tree || typeof tree !== "object") {
            return `
                <div class="rounded-xl border border-dashed border-outline-variant/20 bg-surface-container-high px-3 py-3 text-sm text-on-surface-variant">
                    当前没有可查看的子任务图谱。
                </div>
            `;
        }

        const layout = buildDagLayout(tree);
        if (!layout) {
            return `
                <div class="rounded-xl border border-dashed border-outline-variant/20 bg-surface-container-high px-3 py-3 text-sm text-on-surface-variant">
                    当前没有可查看的子任务图谱。
                </div>
            `;
        }

        const totalNodes = countNodes(tree);
        const childRunCount = Math.max(0, totalNodes - 1);
        const leafRunCount = layout.nodes.filter((node) => node.childCount === 0).length;

        return `
            <div class="space-y-3">
                <div class="flex flex-wrap items-center gap-2 text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant">
                    <span class="rounded-full border border-primary/20 bg-primary/10 px-2.5 py-1 text-primary">${escapeHtml(
                        `${childRunCount} 个子任务`
                    )}</span>
                    <span class="rounded-full border border-outline-variant/15 bg-surface px-2.5 py-1">${escapeHtml(
                        `${layout.maxDepth + 1} 层`
                    )}</span>
                    <span class="rounded-full border border-outline-variant/15 bg-surface px-2.5 py-1">${escapeHtml(
                        `${leafRunCount} 个叶子节点`
                    )}</span>
                </div>
                <div class="rounded-xl border border-outline-variant/10 bg-[radial-gradient(circle_at_top,rgba(0,238,252,0.09),rgba(8,12,18,0.96)_65%)] p-3">
                    <div class="mb-3 flex items-center justify-between gap-3">
                        <div class="text-[11px] leading-relaxed text-on-surface-variant">
                            图中展示父子任务的委派关系。点击节点可切换到对应任务详情。
                        </div>
                        <div class="hidden text-[10px] font-mono uppercase tracking-[0.18em] text-on-surface-variant sm:block">
                            Root → Child
                        </div>
                    </div>
                    <div class="overflow-x-auto overflow-y-hidden pb-1">
                        <div class="relative" style="width:${layout.width}px;height:${layout.height}px">
                            <svg
                                class="absolute inset-0 h-full w-full"
                                viewBox="0 0 ${layout.width} ${layout.height}"
                                preserveAspectRatio="xMinYMin meet"
                                aria-hidden="true"
                            >
                                <defs>
                                    <pattern id="run-dag-grid" width="28" height="28" patternUnits="userSpaceOnUse">
                                        <path d="M 28 0 L 0 0 0 28" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="1"></path>
                                    </pattern>
                                </defs>
                                <rect width="${layout.width}" height="${layout.height}" fill="url(#run-dag-grid)" opacity="0.55"></rect>
                                ${layout.edges.map((edge, index) => renderEdge(edge, index)).join("")}
                            </svg>
                            ${layout.nodes.map((node) => renderNode(node, selectedRunId)).join("")}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    window.RunDagView = {
        renderSection,
    };
})();

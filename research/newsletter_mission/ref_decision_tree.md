# Agentic Design Pattern — Five-Question Decision Tree

**Source:** [machinelearningmastery.com/choosing-the-right-agentic-design-pattern-a-decision-tree-approach](https://machinelearningmastery.com/choosing-the-right-agentic-design-pattern-a-decision-tree-approach/) (fetched via geekfence mirror 2026-05-15, original gave 403 to WebFetch).

## Q1. Is the solution path known in advance?
- YES → Q2a
- NO → Q2b

## Q2a. Is this a fixed workflow?
- Recommendation: **Sequential Workflow Pattern**
- *"Use the model only for tasks like interpretation or generation, while deterministic code handles everything else."*
- Avoid ReAct overkill when steps are pre-defined.

## Q2b. Does the task require tools or external info?
- Almost always YES → **Tool-Use Pattern** (foundational)
- *"Tool use sits beneath reasoning patterns (ReAct, Planning) rather than replacing them."*

## Q3. Is the task structure articulable before execution begins?
- YES → **Planning with ReAct inside steps**
  *"Planning exposes dependencies early and avoids mid-execution surprises."*
- NO → **ReAct Pattern** (continue to Q4)

## Q4. Does output quality matter more than response speed?
- YES → **Add Reflection** (generate → critique → refine)
- NO → continue to Q5

## Q5. Specialization OR scale problem one agent can't handle?
- YES → **Multi-Agent Specialist System**
- NO → **Single Agent + Tools + ReAct**
- *"If neither applies, a single strong agent is usually enough; the overhead of multiple agents outweighs the benefit."*

## Pattern destinations
1. Single Agent + Tools + ReAct — default for unknown-path tasks
2. Planning Agent + ReAct Execution — knowable structure
3. Single Agent + Reflection — high-quality output required
4. Multi-Agent Specialist System — specialization OR scale issue

## How OpenMark newsletter mission scores
- Q1: PARTIAL. 5 known formats; topic varies. → Treat per-format as Q2a.
- Q2a: YES. Each newsletter skill has a defined tool sequence. → **Sequential Workflow per format**.
- Q2b: YES. Tools required (semantic search, web fetch, humanizer).
- Q3: YES. Each format has a template, citations rule, voice rules. → **Planning + ReAct**.
- Q4: YES. "Best 10x output" — quality over speed. → **+ Reflection**.
- Q5: YES. Researcher + Composer + Humanizer + Verifier are different expertise. → **Multi-Agent**.

## Resulting pattern stack for OpenMark
**Planner → Multi-Agent Specialist (parallel where independent) → Reflection (verifier)**

Implemented via LangChain's `deepagents` package — see `ref_deepagents.md`.

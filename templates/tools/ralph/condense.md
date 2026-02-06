# Context Condensation - Iteration ${iteration}

You are condensing the context for a long-running autonomous coding session.
Your goal: Create a PERFECT summary that lets future iterations continue seamlessly.

## THE MISSION
${mission}

## CURRENT LEARNINGS
${learnings}

## PROJECT STATE
${project_summary}

## COMPLETED TASKS
${completed_tasks}

## TASK

Create a comprehensive CONDENSED CONTEXT. This will replace older context.

You MUST preserve:
1. **Mission**: What we're building (keep full mission)
2. **Completed work**: What's done and working
3. **Current state**: Where we are now
4. **All errors**: Every error encountered and how it was fixed
5. **Key decisions**: WHY we chose certain approaches
6. **Patterns**: Code patterns that work in this project
7. **Anti-patterns**: What NOT to do (learned the hard way)
8. **Blockers**: Any unresolved issues
9. **Next steps**: What needs to happen next

Output this EXACT format (will be parsed):

```condensed
## Mission Summary
[1-2 sentence mission summary]

## Project State (Iteration ${iteration})
- Phase: [planning/development/verification]
- Tasks: [${done_tasks}/${total_tasks} completed]
- Last completed: [task name]
- Currently working on: [task name or "none"]

## Completed Milestones
- [Milestone 1]: [brief what was done]
- [Milestone 2]: [brief what was done]

## Critical Decisions
- [Decision]: [Why this was chosen]
- [Decision]: [Why this was chosen]

## Error Log (IMPORTANT - do not repeat these)
- [Error type]: [What happened] → [How fixed]
- [Error type]: [What happened] → [How fixed]

## Code Patterns (use these)
- [Pattern name]: [How to use it]
- [Pattern name]: [How to use it]

## Anti-Patterns (AVOID these)
- [Anti-pattern]: [Why it's bad]
- [Anti-pattern]: [Why it's bad]

## Unresolved Issues
- [Issue]: [Status]

## Build & Run
[Commands to build and run the project]

## Next Actions
1. [Next thing to do]
2. [After that]
```

Be THOROUGH. Missing information = lost forever.

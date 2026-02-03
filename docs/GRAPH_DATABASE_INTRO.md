# Neo4j Graph Database - Introduction

This document provides an overview of the Neo4j Graph Database component in the BTC_ACE trading system. It explains the core concepts, data model, and how graph technology supports the trading reflection and decision-making process.

---

## 1. Core Objective

The graph database serves as a **traceable, explainable, and iterative knowledge graph** for trading decisions. It records:

- **Why a trade was made** (Pattern definition)
- **How the trade result turned out** (Win/Loss verification)
- **How similar situations performed** (Attribution analysis)

This enables the system to learn from historical trading patterns and improve future decision-making.

---

## 2. Core Concepts

### 2.1 Node Types

| Node Type | Purpose | Examples |
|-----------|---------|----------|
| **State** | Market micro-state labels | RSI_Oversold, Vol_Spike, Support_Level, Friday_Afternoon |
| **Pattern** | Meaningful combination of States (strategy logic) | Hash: "RSI_Oversold_Support_Level" |
| **Decision** | Trading direction enumeration | LONG, SHORT, NEUTRAL |
| **Outcome** | Trading result enumeration | WIN, LOSS, BREAK_EVEN |
| **Event** | Complete snapshot of a single trade execution | event_id, timestamp, duration, pnl |

### 2.2 Relationship Types

| Relationship | Source → Target | Meaning | Data |
|--------------|-----------------|---------|------|
| **COMPOSED_OF** | Pattern → State | Pattern's logical definition | No weight |
| **SUGGESTS** | Pattern → Decision | Pattern's recommended action | No weight |
| **MATCHES** | Event → Pattern | Trade matched this pattern | `confidence` score |
| **RESULTED_IN** | Event → Outcome | Trade result | No weight |
| **HAS_CONTEXT** | Event → State | Complete market context | No weight |
| **NEXT_REFLECTION** | Event → Event | Time chain between trades | `interval_minutes` |

---

## 3. The Triangular Closed-Loop Structure

The core innovation of this system is the **triangular closed-loop structure**, which connects Patterns, States, Decisions, Events, and Outcomes in a coherent data model.

```
        Pattern (P)
         /      \
        /        \
   COMPOSED_OF  SUGGESTS
      /            \
     /              \
  State (S)      Decision (D)
        \           /
         \ MATCHES /
          \       /
           Event (E)
            / | \
           /  |  \
    RESULTED_IN | HAS_CONTEXT
         /      |       \
      Outcome   |     State(C)
               |
         NEXT_REFLECTION
              |
           Event(E+1)
```

### The Three Edges of the Triangle

| Edge | Reading Perspective | Business Meaning |
|------|---------------------|------------------|
| **Left** (P → State) | Pattern Definition | "What is the strategy logic?" |
| **Right** (P → D) | Pattern Suggestion | "What action does this pattern recommend?" |
| **Bottom** (E → P/O/S) | Event Verification | "How does history validate this pattern?" |

### Why the Triangular Structure?

- **Completeness**: Defines both "why" (left edge) and "result" (bottom edge)
- **Traceability**: From any Event, you can trace back to the Pattern's core logic
- **No Redundancy**: Pattern as a "semi-public" node avoids duplicate storage

---

## 4. Data Flow Overview

### 4.1 Write Flow (After Trade Closure)

```
Trade Closure
    ↓
Reflection Collection
    ├─ core_logic_states  ──────────────────→ Pattern Hash
├─ all_context_states     │
├─ decision                │
└─ outcome                 │
    ↓
store_to_graph_db()
    ├─ Pattern Deduplication (Sort + MD5 Hash)
    ├─ Create or Update Pattern
    ├─ Link Pattern Definition (COMPOSED_OF)
    ├─ Link Pattern Suggestion (SUGGESTS)
    ├─ Create Event Node
    ├─ Establish Triangular Closed-Loop (MATCHES, RESULTED_IN, HAS_CONTEXT)
    └─ Establish Time Chain (NEXT_REFLECTION)
    ↓
Database Storage
```

### 4.2 Read Flow (During Decision Making)

```
New Market Condition
    ↓
Vector Database Query
    ↓
Similar Event IDs
    ↓
query_similar_events_insight()
    ├─ Pattern Analysis
    ├─ Win Rate Calculation
    └─ State Attribution
    ↓
Deep Analysis Result
├─ dominant_pattern (What pattern dominates?)
├─ common_context (What are the soil conditions?)
├─ success_drivers (What are the Alpha factors?)
└─ failure_drivers (What are the Risk factors?)
    ↓
Agent Decision Reference
```

---

## 5. Core Functions

### 5.1 Writing: insert_trade_reflection()

**When to call**: Immediately after Agent closes a position

**Key Parameters**:

- `event_data`: Trade metadata (event_id, timestamp, duration, realized_pnl)
- `core_logic_states`: Core states that triggered the decision (2-5 States)
- `all_context_states`: Complete market context (5-10 States, may include noise)
- `agent_insight`: Pattern recognition result from the agent
- `decision`: Trading direction (LONG/SHORT/NEUTRAL)
- `outcome`: Trading result (WIN/LOSS/BREAK_EVEN)

**What it does**:
1. Deduplicates Patterns using sorted States + MD5 hash
2. Creates or updates Pattern nodes with热度 (weight)
3. Establishes COMPOSED_OF and SUGGESTS relationships
4. Creates Event nodes and links to Pattern, Outcome, and Context States
5. Builds time chains with the previous Event

### 5.2 Reading: query_similar_events_insight()

**When to call**: When Agent needs decision reference for new market conditions

**Input**: List of similar event IDs from the Vector Database

**Output Structure**:

```json
{
    "dominant_pattern": {
        "name": "Oversold_Rebound_HighVolume",
        "definition": ["RSI_Oversold", "Support_Level"],
        "action": "LONG",
        "baseline_win_rate": 0.67,
        "match_count": 3
    },
    "attribution_analysis": {
        "common_context": [
            {"state": "Bull_Sentiment", "prevalence": 0.75}
        ],
        "success_drivers": [
            {"state": "Vol_Spike", "win_rate": 0.85}
        ],
        "failure_drivers": [
            {"state": "Friday_Afternoon", "win_rate": 0.20}
        ]
    }
}
```

---

## 6. Key Concepts Reference

| Concept | Definition | Application Scenario |
|---------|------------|---------------------|
| **Pattern Hash** | Sorted States + MD5 | Automatic deduplication |
| **Baseline Win Rate** | Historical win rate of all trades in a Pattern | Measure strategy effectiveness |
| **Conditional Win Rate** | Win rate when a specific State appears | Identify Alpha/Risk factors |
| **Prevalence** | Frequency of a State in similar events | Distinguish soil vs. differential factors |
| **Confidence** | Match degree between Event and Pattern | Weighted similarity |
| **Interval Minutes** | Time between two Events | Analyze trading frequency |

---

## 7. Attribution Analysis

The system identifies three types of States in similar event groups:

### 7.1 Soil (Common Context)
- **Condition**: prevalence > 60%
- **Meaning**: Background conditions shared by most similar events
- **Insight**: "What market environment are we in?"

### 7.2 Alpha (Success Drivers)
- **Condition**: win_rate > baseline + 15%
- **Meaning**: States that appear more often in winning trades
- **Insight**: "What factors contribute to success?"

### 7.3 Risk (Failure Drivers)
- **Condition**: win_rate < baseline - 15%
- **Meaning**: States that appear more often in losing trades
- **Insight**: "What should we avoid?"

**Note**: States that are part of the Pattern definition itself are excluded from success/failure driver analysis to prevent logical circularity.

---

## 8. Time Chain (NEXT_REFLECTION)

The NEXT_REFLECTION relationship connects consecutive events in chronological order, storing the time interval between trades.

**Applications**:
- Analyze trading frequency
- Detect over-trading (interval too short)
- Analyze "winning streaks" phenomenon
- Calculate recovery time after losses

---

## 9. Performance Characteristics

| Operation | Complexity | Optimization |
|-----------|------------|--------------|
| Pattern Deduplication | O(n log n) | Hash mapping |
| Write Event | O(S + C) | Single-row operation + indexing |
| Query Similar | O(K log K) | Sorting + LIMIT |
| Time Chain | O(log N) | timestamp index |

Where:
- S = core_logic_states count (typically 2-5)
- C = all_context_states count (typically 5-10)
- K = similar event count (typically 5-20)
- N = total events in database

---

## 10. Quick Reference

| Question | Answer |
|----------|--------|
| **When to write?** | After each position closure via `insert_trade_reflection()` |
| **When to read?** | When new market conditions need decision reference via `query_similar_events_insight()` |
| **Data flow?** | Event → Pattern → Decision/Outcome (triangular closed-loop) |
| **Deduplication?** | Automatic via Pattern hash (no manual handling needed) |
| **Attribution principle?** | Compare winner vs. loser State differences (exclude definition itself) |

---

## 11. System Integration

The graph database works together with:

1. **Vector Database (LSTM)**: Provides similar historical event IDs based on market snapshot embeddings
2. **Agent C (Reflector)**: Writes trade reflections after position closure
3. **Agent A/B (Analyzer/Builder)**: Reads analysis results for decision reference

```
┌──────────────────────────┐         ┌──────────────────────────┐
│   Agent C (Reflector)   │         │   Vector DB (LSTM)       │
│  - Execute trades       │         │  - Generate embeddings   │
│  - Reflect on closes    │         │  - Similarity search     │
│  - Record states        │         │  - Return Event IDs      │
└──────────┬───────────────┘         └──────────┬───────────────┘
           │                                    │
           │ insert_trade_reflection()          │ similar_event_ids
           │                                    │
           ▼                                    ▼
     ┌──────────────────────────────────────────────┐
     │  Neo4j Graph Database (Quant Graph Manager)  │
     └──────────┬───────────────────────────────────┘
                │
                │ query_similar_events_insight()
                │
                ▼
         ┌────────────────────────┐
         │  Agent A/B (Reference) │
         │  Make better decisions │
         └────────────────────────┘
```

---

**Document Version**: 1.0
**Last Updated**: 2026-02-02
**Related Documentation**: GRAPH_CHEATSHEET.md, GRAPH_DATABASE_ARCHITECTURE.md, GRAPH_VISUAL_GUIDE.md

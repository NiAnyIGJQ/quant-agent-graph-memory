# QG-ACE (Quant-Graph Adaptive Cognitive Engine)

> An intelligent trading system based on multi-agent collaboration, vector database similarity matching, and graph database knowledge attribution

## Table of Contents

- [Overview](#overview)
- [Core Architecture](#core-architecture)
- [Backtest Process](#backtest-process)
- [Data Flow](#data-flow)
- [Agent Collaboration](#agent-collaboration)
- [Database Systems](#database-systems)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [FAQ](#faq)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)

---

## Overview

QG-ACE is a quantitative trading system that combines **deep learning**, **knowledge graphs**, and **large language models**, featuring:

- **Historical Pattern Learning**: Identify similar historical market conditions via vector database (LSTM encoding)
- **Knowledge Graph Attribution**: Analyze success/failure drivers using Neo4j graph database
- **Multi-Agent Decision Chain**: Agent S (State Perception) → Agent A (Deep Analysis) → Agent B (Risk Decision) → Agent C (Reflection Learning)
- **Dynamic Period Adaptation**: Support multi-timeframe analysis from 1-minute to monthly
- **Complete Data Tracking**: Full-process state snapshots and logs from position open to close

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Market Data                                   │
│              OHLCV Time Series (1M/5M/15M/1H/4H/1D/1W/1M)        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                    [Stage 1-2]
                  Vectorization & Similarity Query
                         │
         ┌───────────────┴────────────────┐
         │                                │
    Vector DB Results               Graph DB Results
   (Similar Event IDs)            (Dominant Pattern + Attribution)
         │                                │
         └────────────────┬──────────────┘
                          │
                    [Stage 3-5]
                 Agent S/A/B Decision Chain
                          │
         ┌────────────────┴──────────────┐
         │                               │
      Position Open                  Position Close
   (Save TradeContext)              (Agent C Write to DB)
         │                               │
      Save to                          GraphDB + VectorDB
   position_metadata                  Knowledge Base Update
         │                               │
         └────────────────────────────────┘
                      Historical Learning Loop
```

---

## Backtest Process

### Phase 1: Initialization

```python
# main.py
runner = BacktestRunner(
    agent=qwen_agent,
    data_dir='./data',
    start='2024-01-01',
    end='2024-12-31',
    cash=100000.0,
    leverage=1.0
)

# Load multi-period data: 1m, 15m, 1h, 4h, 1d, 1w, 1M
# Load auxiliary data: funding rate, open interest
# Initialize vector DB and graph DB connections
```

**Key Configuration**:
- `main_tf`: Main trading period (e.g., 1h), determines decision frequency
- `vector_tf`: Vector matching period (e.g., 1h), determines pattern recognition granularity
- `data_load_start`: Data loading start point (12-24 months before backtest start), ensuring sufficient high-period data

### Phase 2: Bar-by-Bar Decision (next() method)

Each bar executes the following steps:

#### Step 1: Data Collection and Preprocessing (`engine.py:1000-1050`)

```python
# Extract current bar OHLCV data
current_bar = {
    'timestamp': '2024-06-01 08:30:00',
    'open': 83500.0,
    'high': 84200.0,
    'low': 83200.0,
    'close': 83992.0,
    'volume': 12500.5
}

# Collect multi-timeframe data (MTF)
mtf_data = {
    '1m': [...],   # Recent 30 bars
    '15m': [...],
    '1h': [...],   # Used for vectorization (30 bars)
    '4h': [...],
    '1d': [...],
    '1w': [...],
    '1M': [...]    # Used for long-term trends (36 bars)
}

# Extract perpetual contract market sentiment
funding_rate = -0.00042      # Funding rate
open_interest = 450000000.0  # Open interest in USD
```

#### Step 2: Vector Database Query (`engine.py:1876-1898`)

```python
# Extract 30-period 1H OHLCV data (or other vector_tf period)
ohlcv_array = ohlcv_df[['Open', 'High', 'Low', 'Close', 'Volume']].values  # (30, 5)

# LSTM encoding + cosine similarity query
similar_vectors = vec_db.query_similar(
    ohlcv_array,
    top_k=5,
    threshold=0.9  # Similarity threshold
)

# Output: Top-5 similar historical event IDs
similar_event_ids = ['20250401_090000', '20250330_140000', ...]
has_similar_pattern = len(similar_event_ids) > 0  # Flag for similar market found
```

**Vector DB Internal Logic** (`vector_database.py:264-315`):
1. LSTM Encoder: `(30, 5) → (128,)` dimension compression
2. Cosine Similarity: Compare with historical vector database
3. Top-K Sorting: Return 5 most similar historical events

#### Step 3: Graph Database Query (`engine.py:1900-1922`)

```python
# Use similar event IDs to query graph database
graph_insight = graph_db.query_similar_events_insight(
    similar_event_ids=['20250401_090000_P', '20250330_140000_P', ...]
)

# Output structure:
{
    'dominant_pattern': {
        'name': 'RSI_Div_Vol_Low',          # Pattern name
        'definition': ['RSI_High', 'Volume_Low'],  # Core DNA
        'baseline_win_rate': 0.65,          # Historical win rate 65%
        'match_count': 3                    # Matched 3 historical events
    },
    'attribution_analysis': {
        'common_context': [...],            # Background consensus (>60% prevalence)
        'success_drivers': [                # Success drivers (win rate significantly above baseline)
            {'state': 'Price_High', 'win_rate': 0.80, 'freq': 2}
        ],
        'failure_drivers': [                # Failure drivers (win rate significantly below baseline)
            {'state': 'Funding_Extreme', 'win_rate': 0.30, 'freq': 1}
        ]
    }
}
```

**Graph DB Internal Logic** (`quant_graph_manager.py:254-469`):
1. **Macro Pattern Query**: Find dominant Pattern in similar events and its core definition
2. **Micro Attribution Analysis**: Calculate conditional win rate for each State
3. **Background Consensus Identification**: Find common features with prevalence > 60%

#### Step 4: Agent Decision Chain (`qwen_agent.py:829-1614`)

##### Agent S - Market State Perception (`qwen_agent.py:267-553`)

```python
# Input: Market data + Graph DB historical Pattern
agent_s_output = agent._agent_s_analyze_with_graph(
    market_context=full_market_context,    # Multi-period market data
    graph_insight=graph_insight,           # Historical Pattern info
    state=agent_state
)

# Output: Structured state classification (Dict format)
{
    "reasoning_trace": "【Reasoning Chain】Detailed analysis...",
    "current_states": {
        "RSI_Overbought_1H": "RSI=75.5, overbought condition",
        "Price_High_4H": "Price at 80% of 4H period high range",
        "MACD_Bullish_Daily": "Daily MACD golden cross, momentum upward",
        "1h_Bull": "1H period in bull market (continuous new highs)",
        "4h_Bull": "4H period in bull market"
    },
    "matched_states": {
        "RSI_High": "Matches RSI_High in historical Pattern"
    },
    "missing_states": {
        "Volume_Low": "Historical Pattern requires low volume, but current volume is normal"
    },
    "novel_states": {
        "Momentum_Strong": "Current momentum is strong, not seen in historical Pattern"
    }
}
```

**Key Features**:
- Multi-period state recognition (15m/1h/4h/1d/1w/1M)
- Bull/Bear/Monkey market determination
- Comparison analysis with historical Pattern

##### Agent A - Deep Logic Analysis (`qwen_agent.py:556-636`)

```python
# Input: Agent S states + Graph DB attribution + Monthly data
agent_a_output = agent._agent_a_analyze_with_structure(agent_a_prompt)

# Output: Pattern recognition and core DNA
{
    "reasoning_trace": "【Deep Analysis】Based on Agent S recognized states...",
    "pattern_name": "Pattern_Bull_HighMomentum_Overbought",  # Without WIN/LOSS
    "pattern_description": "Bull market high momentum overbought pullback Pattern",
    "core_dna_states": [                 # 1-3 core DNA (necessary conditions)
        "4h_Bull",
        "RSI_Overbought_1H"
    ],
    "context_states": [                  # Environment Context (auxiliary conditions)
        "Momentum_Strong",
        "MACD_Bullish_Daily",
        "Price_High_4H"
    ],
    "confidence": 0.75                   # Confidence 75%
}
```

**Core DNA vs Context Separation**:
- **Core DNA**: Remove any one, and the Pattern doesn't hold (e.g., oversold + support level)
- **Environment Context**: Present simultaneously but not necessary, used for subsequent micro attribution analysis

##### Agent B - Risk Decision (`qwen_agent.py:1348-1614`)

```python
# Input: Agent A analysis + Account status + Leverage configuration
agent_b_output = agent_b.decide(agent_b_prompt)

# Output: Standardized trading decision
{
    "action": "LONG",                    # LONG/SHORT/HOLD/CLOSE
    "quantity_pct": 0.3,                 # Target position 30%
    "take_profit_price": 85500.0,        # Take profit price
    "stop_loss_price": 82500.0,          # Stop loss price
    "limit_price": 0.0,                  # Limit price (0 for market)
    "confidence": 0.75,                  # Decision confidence
    "risk_reward_ratio": 2.5,            # Risk-reward ratio
    "reasoning": "Based on 4H bull + 1H overbought...",
    "position_adjustment_reason": ""     # Position adjustment reason
}
```

**Risk Management**:
- Leverage education: Actual leverage = `quantity_pct × leverage`
- Take profit and stop loss must be set (cannot be 0)
- Adjust position holding time expectation based on main trading period

#### Step 5: Position Open Execution and State Saving (`engine.py:2098-2148`)

```python
if decision.action in ['LONG', 'SHORT']:
    # Extract market states (from Agent S only, no longer use hardcoded indicators)
    market_states = agent_s_output.current_states.keys()  # e.g., ['RSI_Overbought_1H', 'Price_High_4H', ...]

    # Sort states to ensure Hash stability
    market_states_sorted = sorted(market_states)

    # Save complete context at position open
    position_metadata = {
        's_output': agent_s_output,           # Agent S complete output object
        'a_output': agent_a_output,           # Agent A complete output object
        'b_output': decision,                 # Agent B decision
        'market_states': market_states_sorted,  # Sorted state list
        'entry_price': 83992.0,
        'entry_time': '2024-06-01T08:30:00',
        'entry_ohlcv': ohlcv_array,           # (30, 5) vector data
        'entry_decision': 'LONG',
        'graph_search_summary': 'Found 3 similar historical patterns, 65% win rate'
    }

    # Execute position open
    self.buy()  # or self.sell()
```

**Similar Pattern Flag** (`SNAPSHOT_SIMILAR_PATTERN_UPDATE.md`):
- Record `similar_pattern_found` field in `time_series_snapshot.csv` and `trade_decisions.csv`
- `YES`: Found similar historical market (vector DB result > 0)
- `NO`: No similar market found
- `N/A`: Not applicable (system orders, close records, etc.)

### Phase 3: Position Close Execution and Knowledge Learning

#### Step 6: Position Close Trigger (`engine.py:974-1086`)

Close methods:
1. **TP Trigger**: Price reaches take_profit_price
2. **SL Trigger**: Price reaches stop_loss_price
3. **Manual Close**: Agent B decides CLOSE

```python
def notify_trade(self, trade):
    # Calculate P&L
    pnl = trade.pnlcomm  # Realized P&L
    outcome = 'WIN' if pnl > 0 else 'LOSS'

    # Create TradeContext black box (freeze all analysis states at position open)
    trade_context = TradeContext(
        event_id='trade_a1b2c3d4_5678',
        entry_timestamp=position_metadata['entry_time'],
        s_output=position_metadata['s_output'],  # Agent S output
        a_output=position_metadata['a_output'],  # Agent A output
        b_output=position_metadata['b_output'],  # Agent B decision
        graph_search_summary=position_metadata['graph_search_summary']
    )
```

#### Step 7: Agent C-1 Vector DB Write (`engine.py:1123-1215`)

```python
# Write 30-period OHLCV data at position open to vector DB
entry_ohlcv = position_metadata['entry_ohlcv']  # (30, 5)
entry_time = position_metadata['entry_time']

vec_db.add_vector(
    ohlcv_vector=entry_ohlcv,
    timestamp=pd.Timestamp(entry_time)
)

# Internal logic:
# 1. LSTM encoding: (30, 5) → (128,)
# 2. Store vector: stored_embeddings.append(embedding)
# 3. Store metadata: event_ids.append(event_id), timestamps.append(timestamp)
```

#### Step 8: Agent C-2 Graph DB Write (`engine.py:1219-1561`)

```python
# Recover market states at position open from position_metadata
core_logic_states = position_metadata['market_states']  # Sorted
trade_direction = position_metadata['entry_decision']  # 'LONG' or 'SHORT'

# Build complete agent_insight (containing full S/A/B info)
agent_insight = {
    'pattern_name': agent_a_output.pattern_name,  # From Agent A
    'description': f'{trade_direction} trade with states: {core_logic_states}',
    'agent_s': {  # Agent S state detection
        'current_states': agent_s_output.current_states,
        'matched_states': agent_s_output.matched_states,
        'missing_states': agent_s_output.missing_states,
        'novel_states': agent_s_output.novel_states
    },
    'agent_a': {  # Agent A pattern analysis
        'pattern_name': agent_a_output.pattern_name,
        'core_dna_states': agent_a_output.core_dna_states,  # Core DNA
        'context_states': agent_a_output.context_states,    # Environment Context
        'confidence': agent_a_output.confidence
    },
    'agent_b': {  # Agent B decision info
        'action': agent_b_output.action,
        'quantity_pct': agent_b_output.quantity_pct,
        'confidence': agent_b_output.confidence
    }
}

# Write to Neo4j graph database
graph_db.insert_trade_reflection(
    event_data={'event_id': event_id, 'timestamp': entry_time, 'duration': 1, 'realized_pnl': pnl},
    core_logic_states=core_logic_states,    # Core logic states (for Pattern Hash)
    all_context_states=core_logic_states,   # Full observation states (for attribution analysis)
    agent_insight=agent_insight,
    decision=trade_direction,
    outcome=outcome,
    match_confidence=int(agent_a_output.confidence * 100)
)
```

**Graph DB Triangular Closed-Loop Structure** (`quant_graph_manager.py:158-230`):

```cypher
// 1. Create/Update Pattern node (based on Hash)
MERGE (p:Pattern {pattern_hash: hash(sorted_states)})
  ON CREATE SET p.name = pattern_name, p.weight = 1.0
  ON MATCH SET p.weight = p.weight + confidence

// 2. Link Pattern definition (COMPOSED_OF)
UNWIND core_states AS state_name
MERGE (s:State {name: state_name})
MERGE (p)-[:COMPOSED_OF]->(s)

// 3. Create Event node (core anchor)
CREATE (e:Event {
    event_id: event_id,
    timestamp: datetime(entry_time),
    duration: duration,
    realized_pnl: pnl
})

// 4. Establish triangular closed-loop
MERGE (e)-[:MATCHES {confidence: conf}]->(p)   // Logic attribution
MERGE (e)-[:RESULTED_IN]->(o:Outcome)          // Result verification

// 5. Link full observation context (for micro attribution)
UNWIND all_context AS ctx_state_name
MERGE (cs:State {name: ctx_state_name})
MERGE (e)-[:HAS_CONTEXT]->(cs)

// 6. Establish time chain
MATCH (last_e:Event) WHERE last_e.timestamp < e.timestamp
WITH e, last_e ORDER BY last_e.timestamp DESC LIMIT 1
MERGE (last_e)-[r:NEXT_REFLECTION]->(e)
```

---

## Data Flow

### Complete Data Flow (10 Key Stages)

```
[Position Open Time T0]
────────────────────────────────────────

Market Data (OHLCV + MTF)
    │
    ├─> [Stage 1] Data Collection: Extract 30-period OHLCV for vectorization
    │
    ├─> [Stage 2] Vector Query: LSTM encoding → Cosine similarity → Top-5 event IDs
    │         └─> similar_event_ids = ['20250401_090000', ...]
    │
    ├─> [Stage 3] Graph DB Query: Pattern analysis + Attribution analysis
    │         └─> graph_insight = {dominant_pattern, attribution_analysis}
    │
    ├─> [Stage 4] Agent S: Market state perception (Dict format states)
    │         └─> agent_s_output = {current_states, matched_states, ...}
    │
    ├─> [Stage 5] Agent A: Deep analysis (Pattern + Core DNA)
    │         └─> agent_a_output = {pattern_name, core_dna_states, ...}
    │
    ├─> [Stage 6] Agent B: Risk decision
    │         └─> decision = {action, quantity_pct, take_profit, stop_loss}
    │
    └─> [Stage 7] Save to position_metadata
              └─> {s_output, a_output, b_output, entry_ohlcv, market_states}


[Position Close Time Tn]
────────────────────────────────────────

Trade Close Signal (TP/SL/MANUAL)
    │
    ├─> [Stage 8] Create TradeContext black box
    │         └─> Recover all Agent outputs from position_metadata
    │
    ├─> [Stage 9] Agent C-1: Vector DB write
    │         └─> vec_db.add_vector(entry_ohlcv, timestamp)
    │
    └─> [Stage 10] Agent C-2: Graph DB write
              └─> graph_db.insert_trade_reflection(...)
                    ├─> Event node creation
                    ├─> Pattern node creation/update (Hash merge)
                    └─> Triangular closed-loop relationship establishment
```

### Data Volume Estimation

| Stage | Data Structure | Size |
|-------|---------------|------|
| 1 | OHLCV (30×5) | 150 floating-point numbers (~1.2KB) |
| 2 | Similar results (5 items) | 5 Event IDs + similarities (~0.5KB) |
| 3 | Graph DB query result | 1 Pattern + 10-15 drivers (~2-3KB) |
| 5 | Agent S/A/B output | 3 structured objects (~5-10KB) |
| 8 | TradeContext | Complete context object (~10KB) |
| 10 | Graph DB write | 1 Event + 8-15 States + 20+ relationships |

---

## Agent Collaboration

### Agent S - Market State Perceptor

**Responsibility**: Multi-period market state recognition and historical Pattern matching

**Input**:
- Multi-period market data (1m/15m/1h/4h/1d/1w/1M)
- Graph DB historical Pattern info
- Perpetual contract market sentiment (funding rate, open interest)

**Output Format** (`output_schemas.py:AgentSOutput`):
```python
{
    "reasoning_trace": str,              # Complete reasoning chain
    "current_states": Dict[str, str],    # {"state_name": "reasoning"}
    "matched_states": Dict[str, str],    # States matching historical Pattern
    "missing_states": Dict[str, str],    # Required by historical Pattern but missing
    "novel_states": Dict[str, str]       # New states unique to current situation
}
```

**State Naming Convention**:
- Format: `Capitalized_Description_Period`
- Example: `RSI_Overbought_4H`, `Volume_Spike_Daily`, `1h_Bull`, `4h_Monkey`
- Bull/Bear/Monkey determination: `{tf}_Bull`, `{tf}_Bear`, `{tf}_Monkey`

### Agent A - Deep Analyst

**Responsibility**: Pattern recognition, core DNA extraction, logic attribution

**Input**:
- Agent S state classification
- Graph DB attribution analysis (success/failure drivers)
- Monthly data (36 periods, for long-term trends)
- Account status and historical operations

**Output Format** (`output_schemas.py:SchemaAgentAOutput`):
```python
{
    "reasoning_trace": str,                  # Deep analysis reasoning chain
    "pattern_name": str,                     # e.g., "Pattern_Bull_HighMomentum"
    "pattern_description": str,              # Pattern description
    "core_dna_states": List[str],            # 1-3 core DNA (necessary conditions)
    "context_states": List[str],             # 0+ environment Context (auxiliary conditions)
    "confidence": float                      # Confidence 0.0-1.0
}
```

**Pattern Naming Constraints**:
- Correct: `Pattern_Reversal_RSI`, `Pattern_Breakout_Volume`
- Incorrect: `Pattern_SHORT_WIN`, `Trade_LONG_LOSS` (cannot contain WIN/LOSS)

**Core DNA vs Context**:
- **Core DNA**: Remove any one, Pattern doesn't hold
- **Environment Context**: Present simultaneously but not necessary, used for micro attribution analysis

### Agent B - Risk Decision Maker

**Responsibility**: Risk assessment, position management, take profit/stop loss setting

**Input**:
- Agent A Pattern analysis
- Current account status (positions, cash, P&L)
- Leverage configuration and main trading period
- Graph DB historical Pattern performance

**Output Format** (`output_schemas.py:SchemaAgentBOutput`):
```python
{
    "action": str,                           # LONG/SHORT/HOLD/CLOSE
    "quantity_pct": float,                   # Target position percentage 0.0-1.0
    "take_profit_price": float,              # Take profit price (must > 0)
    "stop_loss_price": float,                # Stop loss price (must > 0)
    "limit_price": float,                    # Limit price (0 = market)
    "confidence": float,                     # Decision confidence
    "risk_reward_ratio": float,              # Risk-reward ratio
    "reasoning": str,                        # Decision reasoning
    "position_adjustment_reason": str        # Position adjustment reason
}
```

**Decision Principles**:
- Take profit and stop loss must be set (cannot be 0)
- Actual leverage = `quantity_pct × system_leverage`
- Adjust position holding time expectation based on main trading period
- Set stop loss represents maximum acceptable loss, should not actively close

### Agent C - Reflection Learner

**Responsibility**: Knowledge consolidation and database write after trade close

**C-1: Vector DB Write**
- Encode 30-period OHLCV data at position open into 128-dimensional vector
- Store in vector DB for subsequent similarity queries

**C-2: Graph DB Write**
- Create Event node (trade record)
- Create/Update Pattern node (based on Hash merge)
- Establish triangular closed-loop: Event → Pattern → Outcome
- Link full Context states (for micro attribution)

---

## Database Systems

### Vector Database (Vector DB)

**Tech Stack**: LSTM Encoder + Cosine Similarity

**Data Flow**:
```python
# Write
OHLCV (30, 5) → LSTM Encoder → Embedding (128,) → Storage

# Query
Query OHLCV → LSTM Encoder → Embedding (128,)
           → Cosine similarity calculation → Top-K similar Event ID
```

**Core Methods** (`vector_database.py`):
- `add_vector(ohlcv_vector, timestamp)`: Add new vector
- `query_similar(ohlcv_vector, top_k=5, threshold=0.9)`: Query similar vectors

### Graph Database (Graph DB)

**Tech Stack**: Neo4j + Cypher Query

**Core Nodes**:
- **Event**: Trade event (event_id, timestamp, realized_pnl)
- **Pattern**: Trade pattern (pattern_hash, name, weight)
- **State**: Market state (name)
- **Decision**: Decision type (LONG/SHORT)
- **Outcome**: Trade result (WIN/LOSS)

**Core Relationships**:
- `Event -[:MATCHES]-> Pattern`: Logic attribution
- `Event -[:RESULTED_IN]-> Outcome`: Result verification
- `Event -[:HAS_CONTEXT]-> State`: Full observation
- `Pattern -[:COMPOSED_OF]-> State`: Pattern definition
- `Pattern -[:SUGGESTS]-> Decision`: Suggested action
- `Event -[:NEXT_REFLECTION]-> Event`: Time chain

**Pattern Hash Mechanism** (`quant_graph_manager.py:28-62`):
```python
# Ensure unordered state list generates same Hash
sorted_states = sorted(['RSI_Oversold', 'Support_Hit'])  # Sort
combined_str = "_".join(sorted_states)                   # Concatenate
pattern_hash = md5(combined_str)                         # Hash

# Result: ['A', 'B'] and ['B', 'A'] generate same Hash
```

**Attribution Analysis Query** (`quant_graph_manager.py:254-469`):

1. **Macro Pattern Query**:
```cypher
MATCH (e:Event) WHERE e.event_id IN $similar_ids
MATCH (e)-[:MATCHES]->(p:Pattern)
MATCH (e)-[:RESULTED_IN]->(o:Outcome)
MATCH (p)-[:COMPOSED_OF]->(def_s:State)

RETURN p.name, p.description,
       collect(def_s.name) AS pattern_definition,
       count(e) AS occurrence,
       sum(CASE WHEN o.class='WIN' THEN 1 ELSE 0 END) / count(e) AS win_rate
```

2. **Micro Attribution Query**:
```cypher
MATCH (e:Event) WHERE e.event_id IN $similar_ids
MATCH (e)-[:HAS_CONTEXT]->(s:State)
MATCH (e)-[:RESULTED_IN]->(o:Outcome)

WITH s.name AS state_name,
     count(e) AS total_freq,
     sum(CASE WHEN o.class='WIN' THEN 1 ELSE 0 END) / count(e) AS conditional_win_rate,
     count(e) / size($similar_ids) AS prevalence

RETURN state_name, conditional_win_rate, prevalence
ORDER BY conditional_win_rate DESC
```

**Driver Factor Identification Logic**:
- **Success Drivers (Alpha)**: `win_rate > baseline_win_rate + 0.15`
- **Failure Drivers (Risk)**: `win_rate < baseline_win_rate - 0.15` or `win_rate < 0.33`
- **Background Consensus (Context)**: `prevalence >= 0.6`

---

## Quick Start

### 1. Environment Configuration

```bash
# Install dependencies
pip install -r requirements.txt

# Configure Neo4j database
# Modify connection info in ace_trading/config.py
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"
```

### 2. Data Preparation

Place the following data files in `data/` directory:
- `BTC-USDT_1m_2024_2025.csv`
- `BTC-USDT_15m_2021_2025.csv`
- `BTC-USDT_1H_2021_2025.csv`
- `BTC-USDT_4H_2021_2025.csv`
- `BTC-USDT_1D_2021_2025.csv`
- `BTC-USDT_1W_2021_2025.csv`
- `BTC-USDT_1mon_2021_2025.csv`
- `BTC-USDT-SWAP_FundingRate.csv`
- `BTC-USDT-SWAP_OpenInterest_4H.csv`

### 3. Run Backtest

```python
# main.py
from ace_trading.engine import BacktestRunner
from ace_trading.agents.qwen_agent import QwenAgent

# Initialize Agent
agent = QwenAgent()

# Configure backtest
runner = BacktestRunner(
    agent=agent,
    data_dir='./data',
    start='2024-01-01',
    end='2024-12-31',
    cash=100000.0,
    commission=0.001,
    leverage=1.0
)

# Start backtest (main_tf=main trading period, vector_tf=vector matching period)
results, cerebro, snapshot_dir = runner.run(
    start_backtest=True,
    main_tf='1h',      # Main trading period: 1 hour
    vector_tf='1h'     # Vector matching period: 1 hour
)

# Save results
runner.save_snapshot(cerebro, results[0], out_dir=snapshot_dir)
```

### 4. View Results

After backtest completion, generate in `snapshots/<timestamp>/`:
- `analyzers.json`: Performance metrics (Sharpe ratio, max drawdown, etc.)
- `meta.json`: Backtest metadata (starting capital, leverage, etc.)
- `time_series_snapshot.csv`: Bar-by-bar NAV snapshot
- `trade_decisions.csv`: All trading decision logs
- `history_log.txt`: Agent historical operation records
- `daily_nav.csv` / `weekly_nav.csv` / `monthly_nav.csv`: Periodic NAV summary
- `vector_db_summary.txt`: Vector DB summary
- `prompt_logs/<agent>_prompts.log`: Agent prompt logs

**Key Field Descriptions**:

`time_series_snapshot.csv`:
- `similar_pattern_found`: YES (similar market found) / NO (not found) / N/A (not applicable)
- `decision_type`: LONG / SHORT / HOLD / CLOSE_TP_HIT / CLOSE_SL_HIT / CLOSE_MANUAL

`trade_decisions.csv`:
- `decision_type`: Agent decision type
- `order_status`: PENDING / FILLED / FAILED
- `similar_pattern_found`: Whether matched similar historical market

---

## Key Features and Advantages

### 1. Historical Pattern Learning

- **Vector Database**: LSTM encoding 30-period OHLCV → 128-dim vector → cosine similarity matching
- **Similarity Threshold**: Default 0.9, adjustable
- **Auto Learning**: Auto write to vector DB on each close, continuously accumulate historical patterns

### 2. Knowledge Graph Attribution

- **Triangular Closed-Loop Topology**: Event → Pattern → Outcome, clear and traceable logic
- **Automatic Pattern Merge**: Based on Hash mechanism, same state combinations automatically merge to same Pattern
- **Micro Attribution Analysis**: Identify success drivers (Alpha) and failure drivers (Risk)
- **Background Consensus Identification**: Find common features with high prevalence (>60%)

### 3. Multi-Agent Collaboration

- **State Perception (Agent S)**: Multi-period state recognition + historical Pattern matching
- **Deep Analysis (Agent A)**: Pattern recognition + core DNA extraction
- **Risk Decision (Agent B)**: Position management + take profit/stop loss setting
- **Reflection Learning (Agent C)**: Knowledge consolidation + database write

### 4. Complete Data Tracking

- **TradeContext Black Box**: Freeze all analysis states at position open
- **Prompt Logs**: Record complete input/output for each Agent (`prompt_logs/`)
- **Token Statistics**: Record Token consumption for each LLM call
- **Decision Logs**: Real-time record of all decisions and order status

### 5. Multi-Period Adaptation

- **Flexible Configuration**: Support any main trading period from 1m to 1M
- **Independent Vector Period**: Vector matching period configurable independently
- **Data Pre-warm**: Auto load data 12-24 months in advance, ensuring sufficient high-period data

---

## FAQ

### Q1: Why separate core_dna and context_states?

**A**:
- **Core DNA** (1-3): Necessary conditions defining Pattern, used to calculate Pattern Hash and merge same logic Patterns
- **Environment Context** (0+): Auxiliary conditions, used for micro attribution analysis, explaining why same Pattern sometimes wins and sometimes loses

For example:
- Pattern = "Oversold Rebound" → Core DNA = `['RSI_Oversold', 'Support_Hit']`
- Success Context = `['Volume_Spike', 'Bull_Sentiment']`
- Failure Context = `['Friday_Afternoon', 'Low_Liquidity']`

### Q2: Why can't Pattern name contain WIN/LOSS?

**A**:
- Pattern represents **market conditions and logic signals**, unrelated to trade results
- Same Pattern (e.g., "Oversold Rebound") can lead to WIN or LOSS
- Trade results recorded via graph database `Event -[:RESULTED_IN]-> Outcome` relationship
- Mixing Outcome and Pattern destroys pattern learning and attribution analysis

### Q3: Why sort states?

**A**:
- **Hash Stability**: Ensure `['RSI_Oversold', 'Support_Hit']` and `['Support_Hit', 'RSI_Oversold']` generate same Pattern Hash
- **Pattern Merge**: Same state combinations (regardless of order) automatically merge to same Pattern, improve statistical validity
- **Agent S Output Standardization**: Correctly identify Pattern regardless of order LLM returns states

### Q4: What is similar_pattern_found for?

**A**:
- **Backtest Analysis**: Statistics on win rate difference between trades with/without similar historical patterns
- **Strategy Optimization**: Identify which market states are easier to find historical references
- **Risk Control**: When no historical reference, reduce position or increase risk threshold
- **Pattern Learning**: Analyze similar pattern trade results, improve Agent decision logic

### Q5: How to adjust vector DB similarity threshold?

**A**:
Modify in `engine.py:1886`:
```python
similar_vectors = self.params.vec_db.query_similar(
    ohlcv_array,
    top_k=5,
    threshold=0.9  # Lower threshold (e.g., 0.7) finds more similar markets but lower similarity
)
```

Suggestions:
- `threshold=0.9`: Strict match, return only very similar historical markets
- `threshold=0.7-0.8`: Medium match, return more historical references
- `threshold<0.7`: Loose match, may introduce noise

---

## Tech Stack

- **Backtest Engine**: Backtrader
- **Deep Learning**: PyTorch (LSTM Encoder)
- **Large Language Model**: Qwen via LangChain
- **Graph Database**: Neo4j + Cypher
- **Data Processing**: Pandas, NumPy
- **Logging System**: Custom prompt logs + Token statistics

---

## Project Structure

```
QG-ACE/
├── ace_trading/
│   ├── engine.py                    # Core backtest engine (10 data flow stages)
│   ├── framework.py                 # TradeContext, AgentOutput data structures
│   ├── agents/
│   │   ├── qwen_agent.py            # QwenAgent (S/A/B decision chain)
│   │   └── output_schemas.py        # Structured output models (Pydantic)
│   ├── graph/
│   │   └── quant_graph_manager.py   # Graph database manager (Pattern attribution)
│   ├── LSTM/
│   │   ├── vector_database.py       # Vector database (LSTM encoding)
│   │   └── vector_db_manager.py     # Vector DB persistence management
│   ├── prompt_logger.py             # Prompt logging system
│   ├── token_logger.py              # Token statistics system
│   └── config.py                    # Configuration file
├── data/                            # Market data (CSV files)
├── snapshots/                       # Backtest result snapshots
├── prompt_logs/                     # Agent prompt logs
├── main.py                          # Startup script
├── docs/
│   ├── 数据流传输手册.md             # Complete data flow documentation (Chinese)
│   ├── DATA_FLOW.md                 # Complete data flow documentation (English)
│   ├── GRAPH_DATABASE_INTRO.md      # Graph database introduction
│   ├── GRAPH_CHEATSHEET.md          # Graph database cheatsheet
│   ├── GRAPH_DATABASE_ARCHITECTURE.md  # Graph database architecture
│   ├── GRAPH_VISUAL_GUIDE.md        # Graph database visual guide
│   └── SNAPSHOT_SIMILAR_PATTERN_UPDATE.md  # Similar pattern field description
├── pyproject.toml                   # Project configuration
├── uv.lock                          # UV lock file
└── README.md                        # This file
```

---

## License

MIT License

---

## Contact

For questions or suggestions, please contact:

- Project Repository: [GitHub Issues](https://github.com/your-repo/qg-ace/issues)
- Email: your-email@example.com

---

**Disclaimer**: This system is for learning and research purposes only and does not constitute any investment advice. Quantitative trading involves risks, please use with caution.

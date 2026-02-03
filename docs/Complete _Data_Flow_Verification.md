# Complete Data Flow Verification

**Date**: 2025-12-22
**Scope**: From market data input → similarity matching → graph database indexing → Agent data reception
**Objective**: Precisely track data structures and transformation processes at each stage, no code modifications, documentation only
**Verification Result**: Complete end-to端 data flow channel confirmed

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Detailed Data Flow (10 Key Stages)](#detailed-data-flow-10-key-stages)
3. [Data Structure Reference](#data-structure-reference)
4. [Error Handling and Degradation](#error-handling-and-degradation)
5. [Complete Flow Visualization](#complete-flow-visualization)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Market Data                                  │
│              OHLCV Time Series (1M/5M/15M/1H...)                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                    [Stage 1-2]
                  Vectorization & Similarity Query
                         │
         ┌───────────────┴────────────────┐
         │                                │
    Vector DB Results               Graph DB Results
   (Similar Event IDs)            (Dominant Pattern)
         │                                │
         └────────────────┬──────────────┘
                          │
                    [Stage 3-5]
                 Agent S/A/B Decision Chain
                          │
         ┌────────────────┴──────────────┐
         │                               │
      Position Open                  Position Close
   (STAGE 0-PROTOCOL)          (TradeContext Recovery)
         │                               │
      Save to                         GraphDB Write
   position_metadata                      │
                              ┌──────┴──────┐
                              │             │
                        Historical Pattern Update
                      Event Node Creation
                    Relationship Link Complete
```

---

## Detailed Data Flow (10 Key Stages)

### Stage 1: Market Data Collection and Preprocessing

**Location**: `engine.py:1000-1050` (next() method start)

**Input Data Structure**:
```python
# Market data from Backtrader
market_data = {
    "timestamp": "2025-04-01 08:30:00",  # Bar timestamp
    "open": 83500.0,
    "high": 84200.0,
    "low": 83200.0,
    "close": 83992.0,
    "volume": 12500.5
}

# Multi-timeframe data (1M/5M/15M/1H/4H)
mtf_data = {
    "1m": [{"o": ..., "h": ..., "l": ..., "c": ..., "v": ...}, ...],
    "5m": [...],
    "15m": [...],
    "1h": [{"o": ..., "h": ..., "l": ..., "c": ..., "v": ...}, ...],  # 30 bars
    "4h": [...]
}

# Perpetual contract market data
funding_info = {
    "funding_rate": -0.00042,  # Funding rate
    "open_interest": 450000000.0  # Open interest in USD
}
```

**Processing Logic** (simplified):
```python
def next(self):
    # 1. Get current bar
    current_bar = {
        'timestamp': self.main_data.datetime.datetime(0).isoformat(),
        'open': self.main_data.open[0],
        'high': self.main_data.high[0],
        'low': self.main_data.low[0],
        'close': self.main_data.close[0],
        'volume': self.main_data.volume[0]
    }

    # 2. Collect multi-timeframe data (via DataFeeds)
    mtf_data = {
        '1m': self._get_ohlcv_array(self.data_1m, lookback=30),
        '5m': self._get_ohlcv_array(self.data_5m, lookback=30),
        '15m': self._get_ohlcv_array(self.data_15m, lookback=30),
        '1h': self._get_ohlcv_array(self.data_1h, lookback=30),  # Key: used for vectorization
        '4h': self._get_ohlcv_array(self.data_4h, lookback=30)
    }

    # 3. Extract perpetual contract data (if available)
    funding_rate = self.params.funding_rate_feed.funding[0]  # Latest funding rate
    open_interest = self.params.oi_feed.openinterest[0]  # Latest open interest
```

**Output Data Structure**:
```python
# Data prepared for next stage
prepared_state = {
    "market_data": current_bar,  # Current bar
    "mtf_data": mtf_data,  # Multi-timeframe data
    "funding_rate": funding_rate,
    "open_interest": open_interest
}
```

---

### Stage 2: Vectorization and Similarity Query (Vector DB Stage)

**Location**: `engine.py:1200-1245` (Vector DB query section)

**Input Data Structure** (from Stage 1):
```python
# 1H vectorization input (most important)
ohlcv_1h_30bars = [
    [open_0, high_0, low_0, close_0, volume_0],  # T-29h
    [open_1, high_1, low_1, close_1, volume_1],  # T-28h
    ...
    [open_29, high_29, low_29, close_29, volume_29]  # T (current)
]  # Shape: (30, 5)

# Data range examples
# open:    [83500.0, 83600.0, ..., 83992.0]
# high:    [84200.0, 84300.0, ..., 84050.0]
# low:     [83200.0, 83300.0, ..., 83800.0]
# close:   [83900.0, 83950.0, ..., 84000.0]
# volume:  [12500.5, 13000.2, ..., 14500.8]
```

**Processing Logic** (simplified):
```python
# engine.py:1200-1220
if self.params.vec_db is not None:
    try:
        # 1. Extract 30 1H OHLCV bars from mtf_data['1h']
        ohlcv_array = self._extract_ohlcv_vector(mtf_data['1h'])  # (30, 5)

        # 2. Call vector DB query (LSTM embedding computation + similarity search)
        similar_results = self.params.vec_db.query_similar(
            ohlcv_vector=ohlcv_array,  # (30, 5) numpy array
            top_k=5,  # Return Top-5 similar events
            threshold=0.7  # Similarity threshold
        )

        # 3. Collect similar Event IDs
        similar_event_ids = [r['event_id'] for r in similar_results]
```

**Method Called** (`vector_database.py:264-315`):
```python
def query_similar(self, ohlcv_vector: np.ndarray, top_k: int = 5,
                  threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    LSTM vectorization + similarity query

    Input:
        ohlcv_vector: (30, 5) array
                     [OHLCV for T-29h, T-28h, ..., T]

    Output:
        List[Dict] with:
        - event_id: str (e.g., "20250401_090000")
        - timestamp: pd.Timestamp
        - similarity: float (0.0-1.0, cosine similarity)
        - rank: int (1-5)
    """

    # 1. LSTM Encoder processing
    # Input shape: (1, 30, 5)  # batch_size=1
    with torch.no_grad():
        embedding = self.lstm_encoder(torch.FloatTensor(ohlcv_vector).unsqueeze(0))
    # Output shape: embedding = (1, 128)  # 128-dim representation

    # 2. Cosine similarity calculation (compare with all saved vectors)
    # Historical vectors in database: shape (num_history, 128)
    similarities = cosine_similarity(embedding, self.stored_embeddings)
    # similarities shape: (1, num_history)

    # 3. Filter + Top-K sorting
    valid_indices = np.where(similarities[0] >= threshold)[0]
    top_indices = np.argsort(similarities[0][valid_indices])[-top_k:][::-1]

    # 4. Return results list
    results = []
    for rank, idx in enumerate(top_indices, 1):
        results.append({
            'event_id': self.event_ids[idx],  # "20250401_090000"
            'timestamp': self.timestamps[idx],
            'similarity': float(similarities[0][idx]),
            'rank': rank
        })
    return results  # List[Dict], usually length 1-5
```

**Output Data Structure**:
```python
similar_results = [
    {
        'event_id': '20250401_090000',  # Historical event ID
        'timestamp': Timestamp('2025-04-01 09:00:00'),
        'similarity': 0.87,
        'rank': 1
    },
    {
        'event_id': '20250330_140000',
        'timestamp': Timestamp('2025-03-30 14:00:00'),
        'similarity': 0.85,
        'rank': 2
    },
    ...  # Maximum 5
]

similar_event_ids = ['20250401_090000', '20250330_140000', ...]
```

**Data Volume**:
- Input: 30×5 = 150 floating-point numbers
- Output: 5 similar event records (each contains event_id + similarity + rank)

---

### Stage 3: Graph Database Query (Graph DB Query Stage)

**Location**: `engine.py:1221-1245` (Graph DB query section)

**Input Data Structure** (from Stage 2):
```python
similar_event_ids = ['20250401_090000', '20250330_140000', '20250320_110000']
```

**Processing Logic** (simplified):
```python
# engine.py:1221-1245
if self.params.graph_db is not None:
    try:
        # Call graph database deep analysis interface
        graph_insight = self.params.graph_db.query_similar_events_insight(
            similar_event_ids=similar_event_ids  # List[str]
        )

        # Safely check return value
        if graph_insight and 'dominant_pattern' in graph_insight:
            dominant_pattern = graph_insight['dominant_pattern']
            # dominant_pattern could be None or Dict
```

**Method Called** (`quant_graph_manager.py:216-425`):

Core logic divided into two Cypher queries:

**Query 1: Macro Pattern Analysis**
```cypher
MATCH (e:Event) WHERE e.event_id IN $ids
MATCH (e)-[:MATCHES]->(p:Pattern)
MATCH (e)-[:RESULTED_IN]->(o:Outcome)
MATCH (p)-[:SUGGESTS]->(d:Decision)

// Aggregate statistics: find dominant pattern
WITH p, d, count(e) AS occurrence,
     sum(CASE WHEN o.class = 'WIN' THEN 1 ELSE 0 END) * 1.0 / count(e) AS win_rate
ORDER BY occurrence DESC LIMIT 1

// Get core definition States for that pattern
MATCH (p)-[:COMPOSED_OF]->(def_s:State)

RETURN
    p.name AS pattern_name,           # Example: "RSI_Div_Vol_Low"
    p.description AS description,
    d.action AS suggested_action,     # "LONG" / "SHORT" / "NEUTRAL"
    occurrence,                       # Times this pattern appears in similar group
    win_rate,                         # Baseline win rate for this pattern
    collect(def_s.name) AS pattern_definition  # ["RSI_High", "Vol_Low"]
```

**Query 2: Micro Attribution Analysis**
```cypher
MATCH (e:Event) WHERE e.event_id IN $ids
MATCH (e)-[:RESULTED_IN]->(o:Outcome)
MATCH (e)-[:HAS_CONTEXT]->(s:State)

// Aggregate performance for each State
WITH s.name AS state_name,
     count(e) AS total_freq,
     sum(CASE WHEN o.class = 'WIN' THEN 1 ELSE 0 END) AS win_count,
     sum(CASE WHEN o.class = 'LOSS' THEN 1 ELSE 0 END) AS loss_count,
     count(e) * 1.0 / size($ids) AS prevalence

// Calculate conditional win rate
WITH state_name, total_freq, win_count, loss_count, prevalence,
     (win_count * 1.0 / total_freq) AS conditional_win_rate

WHERE total_freq >= 2

RETURN
    state_name,
    total_freq,
    win_count,
    loss_count,
    conditional_win_rate,
    prevalence
ORDER BY conditional_win_rate DESC
```

**Output Data Structure**:
```python
graph_insight = {
    'dominant_pattern': {
        "name": "RSI_Div_Vol_Low",              # Pattern name
        "description": "RSI divergence with low volume",
        "definition": ["RSI_High", "Volume_Low"],  # Core definition of this pattern
        "action": "SHORT",                      # Suggested action
        "baseline_win_rate": 0.65,              # 65% win rate
        "match_count": 3                        # Found 3 similar historical events
    },
    'attribution_analysis': {
        'common_context': [
            {
                'state': 'Momentum_Strong',
                'prevalence': 0.67,  # 67% of similar events have this state
                'insight': '...'
            },
            ...
        ],
        'success_drivers': [
            {
                'state': 'Price_High',
                'win_rate': 0.80,  # Win rate is 80% when this state appears
                'freq': 2,
                'insight': '...'
            },
            ...
        ],
        'failure_drivers': [
            {
                'state': 'Funding_Extreme',
                'win_rate': 0.30,  # Win rate is only 30% when this state appears
                'freq': 1,
                'insight': '...'
            },
            ...
        ]
    }
}
```

**Data Volume**:
- Input: 3-5 Event IDs
- Output: 1 dominant pattern + 3 categories of driver factor lists

---

### Stage 4: Agent Call Data Preparation

**Location**: `engine.py:1250-1280` (Before Agent call)

**Input Data Structure** (from Stage 1-3):
```python
# Stage 1 output
current_bar = {...}
mtf_data = {...}
funding_rate = -0.00042
open_interest = 450000000.0

# Stage 2 output
similar_results = [...]
similar_event_ids = [...]

# Stage 3 output
graph_insight = {
    'dominant_pattern': {...} or None,
    'attribution_analysis': {...}
}
```

**Processing Logic** (simplified):
```python
# engine.py:1250-1280
# Prepare context for Agent call
agent_context = {
    'current_bar': current_bar,
    'mtf_data': mtf_data,
    'funding_rate': funding_rate,
    'open_interest': open_interest,
    'similar_event_ids': similar_event_ids,
    'graph_insight': graph_insight,
    'history_logs': self.history_log  # Historical trading log
}

# Save graph database search summary (for TradeContext)
self.position_metadata['graph_search_summary'] = self._summarize_graph_insight(graph_insight)
```

**Output Data Structure**:
```python
agent_context = {
    "market_state": {
        "current_bar": {...},
        "mtf_data": {...},
        "funding_rate": float,
        "open_interest": float
    },
    "historical_context": {
        "similar_event_ids": List[str],  # 3-5 items
        "graph_insight": {
            "dominant_pattern": {...} or None,
            "attribution_analysis": {...}
        }
    },
    "trading_history": [...]  # Historical log
}
```

---

### Stage 5: Agent S/A/B Decision Chain Execution (STAGE 0-PROTOCOL)

**Location**: `engine.py:1285-1350` (Agent call + output save)

**Input Data Structure** (from Stage 4):
```python
# Agent.decide() input
state = {
    'mtf_data': mtf_data,
    'funding_rate': funding_rate,
    'open_interest': open_interest,
    'graph_insight': graph_insight,
    'history_logs': self.history_log
}
```

**Agent Internal Flow**:

#### Agent S Execution (`qwen_agent.py:606-750`)

**Step 1: Generate Structured Output** (`_agent_s_generate_output()`)

**Agent S Output Data Structure** (`StateComparisonResult`):
```python
@dataclass
class StateComparisonResult:
    current_states: List[str]        # ['Bullish', 'RSI_Overbought', 'MACD_Bullish']
    matched_states: List[str]        # States overlapping with historical pattern
    missing_states: List[str]        # States in pattern definition but currently missing
    novel_states: List[str]          # Newly identified states

@dataclass
class AgentSOutput:
    reasoning_trace: str             # Complete reasoning chain log
    data: StateComparisonResult      # Structured state data
```

#### Agent A Execution (`qwen_agent.py:623-853`)

**Step 2: Logic Attribution and Deep Analysis**

**Agent A Output Data Structure** (`LogicDefinition`):
```python
@dataclass
class LogicDefinition:
    core_logic_states: List[str]      # ['Bullish', 'RSI_Overbought']
    pattern_name: str                 # "Bullish_Momentum"
    pattern_description: str          # "Strong uptrend with overbought state..."
    confidence: float                 # 0.75 (confidence level)

@dataclass
class AgentAOutput:
    reasoning_trace: str              # Complete reasoning chain
    data: LogicDefinition             # Structured logic definition
```

#### Agent B Execution (Decision Generation)

**Step 3: Risk Assessment and Trading Decision**

**Output Data Structure** (`ExecutionPlan`):
```python
@dataclass
class ExecutionPlan:
    action: str                       # 'LONG' / 'SHORT' / 'NONE'
    quantity_pct: float               # 0.3 (30%)
    take_profit_price: float
    stop_loss_price: float
```

---

### Stage 6: Agent Output Save to position_metadata

**Location**: `engine.py:1260-1280` (Post-position open processing in next())

**Input Data Structure** (from Stage 5):
```python
agent_s_output = AgentSOutput(...)  # Agent S output
agent_a_output = AgentAOutput(...)  # Agent A output
agent_b_output = ExecutionPlan(...) # Agent B output (TradeDecision)
```

**Processing Logic**:
```python
# engine.py:1260-1280
def next(self):
    # ... Agent call ...
    decision = self.params.agent.decide(state)

    # Extract Agent outputs to position_metadata
    agent_s_output_obj = getattr(self.params.agent, '_agent_s_output_obj', None)
    agent_a_output_obj = getattr(self.params.agent, '_agent_a_output_obj', None)

    if decision.action in ['LONG', 'SHORT']:
        # Execute trade
        self.buy() or self.sell()

        # Save to position_metadata
        self.position_metadata = {
            's_output': agent_s_output_obj,           # AgentSOutput
            'a_output': agent_a_output_obj,           # AgentAOutput
            'b_output': decision,                      # ExecutionPlan
            'entry_time': self.main_data.datetime.datetime(0).isoformat(),
            'entry_decision': decision.action,
            'entry_ohlcv': ohlcv_array,  # (30, 5) for vector DB
            'entry_price': self.main_data.close[0],
            'graph_search_summary': graph_summary
        }

        print(f"[POSITION_METADATA] Saved Agent outputs for TradeContext")
```

**Output Data Structure**:
```python
self.position_metadata = {
    's_output': AgentSOutput(...),          # Structured output object
    'a_output': AgentAOutput(...),          # Structured output object
    'b_output': ExecutionPlan(...),         # Trading decision
    'entry_time': "2025-04-01T08:30:00",
    'entry_decision': "LONG",
    'entry_ohlcv': numpy.ndarray(30, 5),   # Key: for vector DB
    'entry_price': 83992.0,
    'graph_search_summary': '...'
}
```

---

### Stage 7: Market State Extraction and Fusion

**Location**: `engine.py:390-503` (`_extract_market_states()` method)

**Input Data Structure** (from Stage 6):
```python
# Data saved in position_metadata
agent_s_output = position_metadata['s_output']  # AgentSOutput

# Current market indicators
rsi_1h = 75.5
price_position = 'NEAR_HIGH'
macd_state = 'BULLISH'
funding_rate = -0.00042
open_interest = 450000000.0
```

**Output Data Structure**:
```python
market_states = {
    'base_states': [
        'RSI_Overbought',
        'Price_High',
        'MACD_Positive',
        'Funding_Neutral',
        'OI_Stable'
    ],
    'additional_states': [
        'Bullish',
        'RSI_Overbought',  # May be duplicated
        'MACD_Bullish',
        'Momentum_Strong'
    ],
    'confidence': {
        'RSI_Overbought': 0.95,
        'Price_High': 0.85,
        'MACD_Positive': 0.8,
        'Funding_Neutral': 0.7,
        'OI_Stable': 0.75,
        'Bullish': 0.8,
        'MACD_Bullish': 0.8,
        'Momentum_Strong': 0.8
    },
    'all_states': [
        'RSI_Overbought',
        'Price_High',
        'MACD_Positive',
        'Funding_Neutral',
        'OI_Stable',
        'Bullish',
        'MACD_Bullish',
        'Momentum_Strong'
    ]  # Complete list after deduplication
}
```

---

### Stage 8: TradeContext Creation at Position Close

**Location**: `engine.py:680-720` (notify_close() method start)

**Input Data Structure** (from Stage 6-7):
```python
# Recover from position_metadata
s_output = self.position_metadata.get('s_output')           # AgentSOutput
a_output = self.position_metadata.get('a_output')           # AgentAOutput
b_output = self.position_metadata.get('b_output')           # ExecutionPlan
graph_search_summary = self.position_metadata.get('graph_search_summary')

# Close position info
trade = order.executed.comm  # Trade object
pnl = trade.pnlcomm          # Realized P&L
dt_str = self.main_data.datetime.datetime(0).isoformat()
```

**TradeContext Data Structure** (`framework.py`):
```python
@dataclass
class TradeContext:
    """Complete trading context information - freezing all analysis states at position open time"""
    event_id: str                          # "trade_a1b2c3d4_5678"
    entry_timestamp: str                   # "2025-04-01T08:30:00"

    # Agent outputs at position open (crossing time boundary)
    s_output: Optional[AgentSOutput]       # Market state analysis
    a_output: Optional[AgentAOutput]       # Logic definition
    b_output: Optional[ExecutionPlan]      # Execution plan

    # Graph database search summary
    graph_search_summary: str              # "Found 3 similar historical patterns, 65% win rate"
```

**Output Data Structure**:
```python
trade_context = TradeContext(
    event_id='trade_a1b2c3d4_5678',
    entry_timestamp='2025-04-01T08:30:00',
    s_output=AgentSOutput(...),     # Complete market analysis
    a_output=AgentAOutput(...),     # Complete logic definition
    b_output=ExecutionPlan(...),    # Trading plan
    graph_search_summary='Found 3 similar patterns with 65% win rate'
)
```

---

### Stage 9: Agent C-1 Vector DB Write

**Location**: `engine.py:724-740` (Vector DB write section)

**Input Data Structure** (from Stage 6-8):
```python
# OHLCV data saved at position open
entry_ohlcv = self.position_metadata.get('entry_ohlcv')  # (30, 5) array
entry_time = self.position_metadata.get('entry_time')    # Timestamp

# P&L info at position close
pnl = -18.07  # Realized P&L
trade_direction = 'LONG' or 'SHORT'
outcome = 'WIN' or 'LOSS' or 'BREAK_EVEN'
```

**Processing Logic**:
```python
# engine.py:724-740
def notify_close(self):
    # ... TradeContext creation ...

    # Agent C-1: Write to vector DB
    if self.params.vec_db is not None:
        try:
            entry_ohlcv = self.position_metadata.get('entry_ohlcv')
            entry_time = self.position_metadata.get('entry_time', '')

            # Validate OHLCV data
            if entry_ohlcv is not None and len(entry_ohlcv) == 30:
                # Convert to Timestamp
                vec_timestamp = pd.Timestamp(entry_time)

                # Call vector DB add_vector method
                self.params.vec_db.add_vector(
                    ohlcv_vector=entry_ohlcv,    # (30, 5) numpy array
                    timestamp=vec_timestamp       # pandas Timestamp
                )

                # Determine trade outcome
                outcome = 'WIN' if pnl > 0 else ('LOSS' if pnl < 0 else 'BREAK_EVEN')
                trade_direction = 'LONG' if self.position_metadata.get('entry_decision') == 'LONG' else 'SHORT'

                print(f"[VECTOR_DB] Trade ({trade_direction} {outcome}) entry pattern added at {entry_time}")
        except Exception as e:
            logger.error(f"Vector DB write failed: {e}")
```

**Vector DB Internal Processing** (`vector_database.py:192-223`):
```python
def add_vector(self, ohlcv_vector: np.ndarray, timestamp: pd.Timestamp) -> None:
    """
    Add new OHLCV vector to database

    Input:
        ohlcv_vector: (30, 5) numpy array
        timestamp: pd.Timestamp
    """

    # 1. LSTM encoding
    # Input shape: (1, 30, 5)
    with torch.no_grad():
        embedding = self.lstm_encoder(torch.FloatTensor(ohlcv_vector).unsqueeze(0))
    # Output shape: (1, 128)

    # 2. Store
    self.stored_embeddings.append(embedding.numpy())      # (1, 128) → stored
    self.event_ids.append(...)                            # Event ID
    self.timestamps.append(timestamp)                     # Timestamp

    # 3. Mark as updated (use new vector on next query)
    self.updated = True
```

**Output Data Structure**:
```python
# Vector DB internal state update
stored_embeddings = [
    ...,
    np.array([[...128 floating point numbers...]])  # New: current trade vector
]

event_ids = [
    ...,
    'trade_a1b2c3d4_5678'  # New: event ID
]

timestamps = [
    ...,
    Timestamp('2025-04-01 08:30:00')  # New: timestamp
]
```

**Data Volume**:
- Input: 30×5 = 150 floating-point numbers
- Processing: LSTM encoded to 128-dim vector
- Storage: 128 floating-point numbers + metadata

---

### Stage 10: Agent C-2 Graph DB Write

**Location**: `engine.py:813-960` (Graph DB write section)

**Input Data Structure** (from Stage 6-8):
```python
# Close position info
event_id = 'trade_a1b2c3d4_5678'
trade_direction = 'LONG'
outcome = 'LOSS'  # or 'WIN', 'BREAK_EVEN'
pnl = -18.07
duration_minutes = 1

# Market states (extracted at position open)
all_market_states = [
    'RSI_Overbought',
    'Price_High',
    'MACD_Positive',
    'Funding_Neutral',
    'OI_Stable',
    'Bullish',
    'MACD_Bullish',
    'Momentum_Strong'
]
base_states = ['RSI_Overbought', 'Price_High', ...]
additional_states = ['Bullish', 'MACD_Bullish', ...]
confidence = {
    'RSI_Overbought': 0.95,
    ...
}

# Trade record details
event_data = {
    'event_id': 'trade_a1b2c3d4_5678',
    'timestamp': '2025-04-01T08:30:00',
    'duration': 1,  # Minutes
    'realized_pnl': -18.07
}

agent_insight = {
    'pattern_name': 'Trade_LONG_LOSS',
    'description': 'LONG trade with states: RSI_Overbought, Price_High, MACD_Positive, ...',
    'base_states': base_states,
    'additional_states': additional_states,
    'confidence': confidence
}
```

**Processing Logic** (`quant_graph_manager.py:104-209`):

**Neo4j Graph Structure (after execution)**:

```
[Nodes Created]
- Decision {action: "LONG"}
- Outcome {class: "LOSS"}
- Pattern {
    pattern_hash: "hash_abc123def456",
    name: "Trade_LONG_LOSS",
    description: "LONG trade with states: ...",
    weight: 1.0,
    created_at: 2025-04-01T08:30:00Z
  }
- State {name: "RSI_Overbought"}
- State {name: "Price_High"}
- State {name: "MACD_Positive"}
- ... More State nodes ...
- Event {
    event_id: "trade_a1b2c3d4_5678",
    timestamp: 2025-04-01T08:30:00Z,
    duration: 1,
    realized_pnl: -18.07
  }
- last_e:Event {
    event_id: "20250401_085900",
    timestamp: 2025-04-01T08:59:00Z,
    ...
  }

[Relationships Created]
Pattern -[:COMPOSED_OF]-> State (8 edges)
    Example: Pattern -[:COMPOSED_OF]-> State {name: "RSI_Overbought"}

Pattern -[:SUGGESTS]-> Decision

Event -[:MATCHES {confidence: 1}]-> Pattern

Event -[:RESULTED_IN]-> Outcome

Event -[:HAS_CONTEXT]-> State (8 edges)
    Example: Event -[:HAS_CONTEXT]-> State {name: "RSI_Overbought"}

last_e -[:NEXT_REFLECTION {interval_minutes: 1}]-> Event
```

**Output Data Structure** (Neo4j log):
```
[GRAPH_DB] Trade 20250401_090000 logged successfully with outcome: LOSS
- Event ID: trade_a1b2c3d4_5678
- Pattern Hash: hash_abc123def456
- Core States: RSI_Overbought, Price_High, MACD_Positive, Funding_Neutral, OI_Stable
- Additional States: Bullish, MACD_Bullish, Momentum_Strong
- Trade Direction: LONG
- Outcome: LOSS
- PnL: -18.07
- Confidence: 1.0
- Status: Recorded and indexed for pattern learning
```

---

## Data Structure Reference

### Core Data Model (Pydantic)

```python
# === Agent Outputs ===

@dataclass
class StateComparisonResult:
    """Agent S structured output - state comparison"""
    current_states: List[str]
    matched_states: List[str]
    missing_states: List[str]
    novel_states: List[str]

@dataclass
class AgentSOutput:
    """Agent S complete output"""
    reasoning_trace: str           # Reasoning chain
    data: StateComparisonResult    # Structured data

@dataclass
class LogicDefinition:
    """Agent A structured output - logic definition"""
    core_logic_states: List[str]
    pattern_name: str
    pattern_description: str
    confidence: float

@dataclass
class AgentAOutput:
    """Agent A complete output"""
    reasoning_trace: str           # Reasoning chain
    data: LogicDefinition          # Structured data

@dataclass
class ExecutionPlan:
    """Agent B output - execution plan"""
    action: str                    # 'LONG' / 'SHORT' / 'NONE'
    quantity_pct: float
    take_profit_price: float
    stop_loss_price: float

@dataclass
class TradeContext:
    """Trading context crossing time boundaries"""
    event_id: str
    entry_timestamp: str
    s_output: Optional[AgentSOutput]
    a_output: Optional[AgentAOutput]
    b_output: Optional[ExecutionPlan]
    graph_search_summary: str
```

### Vector DB Data Format

```python
# Input
ohlcv_vector: np.ndarray  # Shape (30, 5)
                          # 30 bars of [open, high, low, close, volume]

# Encoding
embedding: torch.Tensor   # Shape (1, 128)
                          # LSTM encoder output

# Query results
similarity_result = [
    {
        'event_id': str,
        'timestamp': pd.Timestamp,
        'similarity': float (0.0-1.0),
        'rank': int (1-5)
    },
    ...
]
```

### Graph DB Data Format

```python
# Query input
similar_event_ids: List[str]  # 3-5 event IDs

# Query results
graph_insight = {
    'dominant_pattern': {
        'name': str,
        'description': str,
        'definition': List[str],      # Core definition states
        'action': str,                # 'LONG' / 'SHORT' / 'NEUTRAL'
        'baseline_win_rate': float,   # 0.0-1.0
        'match_count': int
    } | None,
    'attribution_analysis': {
        'common_context': [
            {
                'state': str,
                'prevalence': float,
                'insight': str
            },
            ...
        ],
        'success_drivers': [
            {
                'state': str,
                'win_rate': float,
                'freq': int,
                'insight': str
            },
            ...
        ],
        'failure_drivers': [...]
    }
}
```

---

## Error Handling and Degradation

### 1. Vector DB Query Failure
```python
# Stage 2 degradation
try:
    similar_results = vec_db.query_similar(ohlcv_array)
except Exception as e:
    logger.error(f"Vector DB query failed: {e}")
    similar_event_ids = []  # Empty list
    graph_insight = {
        'dominant_pattern': None,
        'attribution_analysis': {}
    }
```

### 2. Graph DB Query Failure
```python
# Stage 3 degradation
try:
    graph_insight = graph_db.query_similar_events_insight(similar_event_ids)
except Exception as e:
    logger.error(f"Graph DB query failed: {e}")
    graph_insight = {
        'dominant_pattern': None,
        'attribution_analysis': {}
    }
```

### 3. Agent S JSON Parse Failure
```python
# Stage 5 degradation - call text extraction method
try:
    llm_json = json.loads(json_str)
    current_states = llm_json.get('current_states', [])
except json.JSONDecodeError:
    # Fallback: extract keywords from text
    current_states = self._extract_states_from_text(llm_output_text)
    # If still empty, return ['Neutral']
    if not current_states:
        current_states = ['Neutral']
```

### 4. Agent C Vector DB Write Failure
```python
# Stage 9 handling
if entry_ohlcv is not None and len(entry_ohlcv) == 30:
    try:
        vec_db.add_vector(entry_ohlcv, vec_timestamp)
    except Exception as e:
        logger.error(f"Vector DB write failed: {e}")
        # Continue execution, don't interrupt graph DB write
else:
    logger.warning("Invalid OHLCV data, skipping Vector DB write")
```

### 5. Agent C Graph DB Write Failure
```python
# Stage 10 handling
if graph_db is not None:
    try:
        graph_db.insert_trade_reflection(...)
    except Exception as e:
        logger.error(f"Graph DB write failed: {e}")
        # Log error but continue, don't interrupt backtest flow
```

---

## Complete Flow Visualization

```
[Position Open Time T0]
────────────────────────────────────────

Market Data
    │ (current bar + mtf_data[1h] 30 bars)
    ├─────────────────────────────┐
    │ Stage 1: Data Collection   │
    └──────────┬──────────────────┘
               │
        (ohlcv_array)
               │
    ┌──────────▼──────────┐
    │ Stage 2: Vector DB  │
    │ (LSTM embedding)    │
    └──────────┬──────────┘
               │
      (similar_event_ids)
               │
    ┌──────────▼──────────┐
    │ Stage 3: Graph DB   │
    │ (Pattern Analysis)  │
    └──────────┬──────────┘
               │
       (graph_insight)
               │
    ┌──────────▼──────────┐
    │ Stage 4: Data Prep  │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────────────────┐
    │ Stage 5: Agent Decision Chain   │
    │ ├─ Agent S: State Classification│
    │ ├─ Agent A: Logic Definition    │
    │ └─ Agent B: Trading Decision    │
    └──────────┬──────────────────────┘
               │
        (Agent outputs)
               │
    ┌──────────▼──────────┐
    │ Stage 6: Output Save│
    │ position_metadata   │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ Stage 7: State Fusion│
    │ (Multi-layer merge) │
    └──────────┬──────────┘
               │
          (execution)
               │
         ┌─────▼─────┐
         │  LONG/SHORT
         │   Execute │
         └─────┬─────┘
               │
    ┌──────────▼──────────┐
    │ position_metadata   │
    │ Save all states/data│
    └─────────────────────┘


[Position Close Time Tn]
────────────────────────────────────────

Trade Close Signal
    │
    ├─────────────────────────────┐
    │ Stage 8: TradeContext Create│
    │ (Recover position_metadata) │
    └──────────┬──────────────────┘
               │
        (trade_context)
               │
    ┌──────────┴──────────────────┐
    │                             │
┌───▼────────────┐    ┌──────────▼──────┐
│ Stage 9:       │    │ Stage 10:        │
│ Vector DB Write│    │ Graph DB Write   │
│ (VecDB)        │    │ (GraphDB)        │
└────────────────┘    └──────────┬──────┘
                                 │
                        ┌────────▼────────┐
                        │ Neo4j Graph     │
                        │ Update Complete │
                        └─────────────────┘
```

---

## Key Data Transfer Path Verification

### Agent S State Transfer Completeness

```
Agent S LLM Output (text or JSON)
    ↓
_agent_s_generate_output()
    ├─ JSON parse success → StateComparisonResult
    ├─ JSON parse fail → _extract_states_from_text()
    └─ Return AgentSOutput
    ↓
Agent._agent_s_output_obj (save)
    ↓
next() method extraction
    ↓
position_metadata['s_output']
    ↓
_extract_market_states() LAYER 2
    ├─ agent_s_output.data.current_states
    └─ → additional_states
    ↓
position_metadata['additional_states']
    ↓
notify_close() merge to core_logic_states
    ↓
insert_trade_reflection() Write to Graph DB
    ↓
Neo4j: Event -[:HAS_CONTEXT]-> State
```

### Similarity Matching Transfer Completeness

```
Vector DB Query (top-5 similar events)
    ↓
similar_event_ids = ['id1', 'id2', 'id3', 'id4', 'id5']
    ↓
Graph DB Query (Pattern Analysis + Attribution)
    ↓
graph_insight = {
    'dominant_pattern': {...},
    'attribution_analysis': {...}
}
    ↓
Agent A receives (logic definition input)
    ↓
Agent A Output: LogicDefinition
    ↓
position_metadata['a_output']
    ↓
notify_close() TradeContext
    ↓
Historical learning (reference for next similar scenario)
```

---

## Data Volume Estimation

| Stage | Data Structure | Size |
|-------|---------------|------|
| 1 | OHLCV (30×5) | 150 numbers (~1.2KB) |
| 2 | Similar results (5 items) | 5 Event IDs + 5 similarities (~0.5KB) |
| 3 | Graph DB query result | 1 dominant pattern + 10-15 drivers (~2-3KB) |
| 5 | Agent outputs (3) | AgentSOutput + AgentAOutput + ExecutionPlan (~5-10KB) |
| 8 | TradeContext | 6 fields + nested objects (~10KB) |
| 10 | Graph DB write | 1 Event + 8-15 State nodes + 20+ relationships |

---

## Summary

Complete end-to-end data flow verified:

1. **Market Data** → Vectorization (LSTM encoding)
2. **Similarity Query** → Get historical reference
3. **Graph DB Analysis** → Get Pattern insight
4. **Agent Decision** → Generate structured output
5. **State Fusion** → Multi-layer state merge
6. **TradeContext** → Freeze position open state
7. **Vector DB Write** → Save trading pattern
8. **Graph DB Write** → Save complete trading record

Each stage has clear input/output structure, error handling, and data transformation logic. Complete data flow channel verified.

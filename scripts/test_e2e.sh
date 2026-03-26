#!/bin/bash
# End-to-end test script for AgentMCP API
# Usage: ./scripts/test_e2e.sh [BASE_URL]
# Requires: curl, python3 (for JSON parsing)
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
PASS=0
FAIL=0
SKIP=0
TOTAL=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# ---------- helpers ----------

assert_status() {
    local description="$1"
    local method="$2"
    local endpoint="$3"
    local expected_status="$4"
    local data="${5:-}"

    TOTAL=$((TOTAL + 1))
    local url="${BASE_URL}${endpoint}"

    if [ -n "$data" ]; then
        response=$(curl -s -w "\n%{http_code}" -X "$method" "$url" \
            -H "Content-Type: application/json" -d "$data" 2>/dev/null) || true
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" "$url" 2>/dev/null) || true
    fi

    local status_code
    status_code=$(echo "$response" | tail -1)
    local body
    body=$(echo "$response" | sed '$d')

    if [ "$status_code" = "$expected_status" ]; then
        echo -e "${GREEN}PASS${NC} [$status_code] $description"
        PASS=$((PASS + 1))
        LAST_BODY="$body"
        return 0
    else
        echo -e "${RED}FAIL${NC} [$status_code] $description (expected $expected_status)"
        FAIL=$((FAIL + 1))
        LAST_BODY="$body"
        return 1
    fi
}

assert_body_contains() {
    local description="$1"
    local needle="$2"
    local haystack="${3:-$LAST_BODY}"

    TOTAL=$((TOTAL + 1))

    if echo "$haystack" | grep -qi "$needle"; then
        echo -e "${GREEN}PASS${NC}       $description (contains '$needle')"
        PASS=$((PASS + 1))
        return 0
    else
        echo -e "${RED}FAIL${NC}       $description (missing '$needle')"
        FAIL=$((FAIL + 1))
        return 1
    fi
}

skip_test() {
    local description="$1"
    local reason="$2"
    TOTAL=$((TOTAL + 1))
    SKIP=$((SKIP + 1))
    echo -e "${YELLOW}SKIP${NC} $description -- $reason"
}

# ---------- tests ----------

echo "================================================"
echo " AgentMCP E2E Tests"
echo " Target: $BASE_URL"
echo "================================================"
echo ""

LAST_BODY=""

# 1. Health check
echo "--- Health ---"
assert_status "GET /health returns 200" GET "/health" 200 || true
assert_body_contains "Health response contains 'healthy'" "healthy" || true

# 2. Tools listing
echo ""
echo "--- Tools ---"
assert_status "GET /tools returns 200" GET "/tools" 200 || true
assert_body_contains "Tools response contains 'calculator'" "calculator" || true

# 3. Agent chat (calculator)
echo ""
echo "--- Agent Chat ---"
assert_status "POST /agent/chat with math question" POST "/agent/chat" 200 \
    '{"message": "What is 15 * 7?"}' || true
assert_body_contains "Chat response contains '105'" "105" || true

# 4. RAG ingest (skip -- requires file upload)
echo ""
echo "--- RAG ---"
skip_test "POST /rag/ingest" "requires multipart file upload with demo data"

# 5. RAG query
assert_status "POST /rag/query returns 200" POST "/rag/query" 200 \
    '{"query": "test", "top_k": 3}' || true

# 6. Tracing -- extract run_id from agent chat response and test trace endpoint
echo ""
echo "--- Tracing ---"
RUN_ID=$(curl -s -X POST "${BASE_URL}/agent/chat" \
    -H "Content-Type: application/json" \
    -d '{"message": "What is 1+1?"}' 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('metadata',{}).get('run_id',''))" 2>/dev/null)
if [ -n "$RUN_ID" ]; then
    assert_status "GET /agent/trace/{run_id} returns 200" GET "/agent/trace/${RUN_ID}" 200 || true
else
    skip_test "GET /agent/trace/{run_id}" "could not extract run_id from agent chat response"
fi

# ---------- summary ----------

echo ""
echo "================================================"
echo " Results"
echo "================================================"
echo -e " PASSED: ${GREEN}${PASS}${NC} / TOTAL: ${TOTAL}"
echo -e " FAILED: ${RED}${FAIL}${NC}"
echo -e " SKIPPED: ${YELLOW}${SKIP}${NC}"
echo "================================================"

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}SOME TESTS FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}ALL TESTS PASSED${NC}"
    exit 0
fi

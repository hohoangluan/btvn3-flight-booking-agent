# BTVN#3 - Dựng agent đặt vé máy bay bằng LangChain: Harness và so sánh 3 design pattern

## 1. Tổng quan

### 1.1 Đề bài

> Tìm hiểu LangChain, LangGraph → Tạo tool mockup → Viết lớp harness cho Agent này. Nộp `.py` kèm báo cáo.
>
> 1. Cài đặt đủ các lớp harness: ràng buộc là dữ liệu, tiêu chí hoàn thành kiểm bằng code, kiểm quyền, bàn giao;
> 2. Cài đặt Agent với 3 mẫu thiết kế: ReAct, Plan-then-Execute, Lai;
> 3. Đánh giá hiệu quả của Agent với 3 mẫu thiết kế khác nhau.

Link GitHub: https://github.com/hohoangluan/btvn3-flight-booking-agent

| Yêu cầu | Đáp ứng trong bài | Mục |
|---|---|---|
| Tool mockup | 4 tool trên 10 chuyến bay giả | 4 |
| 1. Harness | `Constraints`, `is_done()`, `PERMISSION` + `check()`, `handoff()` | 5 |
| 2. Ba mẫu thiết kế | `react()`, `plan_execute()`, `hybrid()` | 6 |
| 3. Đánh giá | `evaluate.py`: 15 case × 3 pattern × 3 lần | 7, 8 |

### 1.2 Bài toán

Agent nhận yêu cầu đặt vé máy bay bằng ngôn ngữ tự nhiên, ví dụ: *"Đặt 1 vé SGN → DAD ngày 2026-10-07, khởi hành trước 12:00, giá tối đa 2.000.000 VND."* Agent phải tìm chuyến, đặt chỗ, thanh toán và xác nhận kết quả. Hai điểm khó:

- Ràng buộc cứng (tuyến, ngày, giờ khởi hành, giá). Chuyến rẻ nhất có thể bay chiều; chuyến đúng giờ có thể quá đắt.
- Thanh toán tốn tiền và không hoàn tác được, nên cần người duyệt.

### 1.3 Mục tiêu

1. Xây agent đặt vé với một **harness** (lớp kiểm soát bằng code, model không bỏ qua được): ràng buộc là dữ liệu, hoàn thành được kiểm tra bằng code, có phân quyền, có handoff cho người.
2. Cài cùng agent theo **3 pattern**: ReAct, Plan-then-Execute, Hybrid.
3. **Đánh giá** 3 pattern trên cùng một benchmark, đo tỉ lệ đạt kỳ vọng, số tool call, token và độ trễ.

## 2. Cấu hình môi trường

**Hệ thống.** Windows 11, Python 3.13.

**Thư viện** (`requirements.txt`, pin đúng version đã chạy benchmark):

| Package | Version | Dùng cho |
|---|---|---|
| `langchain` | 1.4.3 | `create_agent`, middleware (`wrap_tool_call`, `ModelCallLimitMiddleware`) |
| `langchain-core` | 1.6.6 | prompt, structured output, callback đếm token |
| `langchain-openai` | 1.6.7 | `ChatOpenAI` |
| `langgraph` | 1.2.12 | runtime graph mà `create_agent` biên dịch thành |
| `pydantic` | 2.13.4 | schema `Plan` / `Step` |
| `python-dotenv` | 1.2.2 | đọc `.env` |
| `pandas` | 3.0.3 | tổng hợp kết quả benchmark |
| `pytest` | 9.0.3 | unit test harness |

**Biến môi trường** (file `.env`, đã git-ignore):

| Biến | Ý nghĩa |
|---|---|
| `NINEROUTER_API_KEY` | API key (bắt buộc) |
| `NINEROUTER_URL` | endpoint OpenAI-compatible, mặc định `http://localhost:20128/v1` |
| `NINEROUTER_MODEL` | tên model; benchmark dùng `ag/gemini-3.8-flash-low` |

File mẫu `.env.example` (có sẵn trong repo) liệt kê đủ 3 biến với giá trị placeholder; copy thành `.env` và điền API key.

LLM được gọi qua **9router**, một router cục bộ có endpoint OpenAI-compatible, nên code chỉ cần `ChatOpenAI(base_url=..., api_key=..., model=...)`. Muốn đổi sang nhà cung cấp OpenAI-compatible khác chỉ cần sửa `.env`.

**Cách chạy.**

```bash
pip install -r requirements.txt
pytest -q test_harness.py                          # 13 test, không cần LLM
python flight_agent.py react                       # chạy 1 pattern, hỏi duyệt thanh toán
python evaluate.py --repeats 3                     # benchmark 3 pattern bắt buộc
python evaluate.py --patterns hybrid_replan --repeats 3   # bonus (Phụ lục A)
```

## 3. LangChain và LangGraph

Project dùng LangChain cho: `ChatOpenAI` trỏ tới 9router, `with_structured_output` để lấy `Plan` có kiểu, `create_agent` cho ReAct và bước find của Hybrid, middleware `@wrap_tool_call` (harness), `ModelCallLimitMiddleware(10)`, và `UsageMetadataCallbackHandler` để đếm token.

**LangGraph nằm ở đâu.** `create_agent` của LangChain 1.x được biên dịch thành một `CompiledStateGraph` của LangGraph; ReAct và bước find của Hybrid chạy trên graph này. Graph của agent ReAct:

```mermaid
flowchart TD
    S([__start__]) --> BM["ModelCallLimit.before_model"]
    BM -->|"chưa quá 10"| M["node model<br/>LLM gọi tool hoặc trả lời"]
    BM -->|"quá 10"| E([__end__])
    M --> AM["ModelCallLimit.after_model"]
    AM -->|"có tool_call"| T["node tools<br/>chạy tool qua harness"]
    AM -->|"không có tool_call"| E
    T --> BM
```

**Vì sao Plan-then-Execute và Hybrid không viết bằng `StateGraph`.** Hai pattern này viết bằng Python thuần (một vòng `for` và chuỗi bước cố định). Luồng của chúng là tuyến tính, không có cạnh rẽ nhánh do LLM quyết định, nên một `StateGraph` chỉ bọc lại đúng các bước đó mà không đổi hành vi. Giữ Python thuần giúp so sánh pattern công bằng: khác biệt giữa 3 pattern nằm ở *ai quyết định bước kế tiếp*, không lẫn với khác biệt framework. Cả hai vẫn ánh xạ trực tiếp sang LangGraph (mỗi bước một node, `is_done()` là node cuối) nếu cần.

## 4. Mock Flight Environment

### 4.1 Dữ liệu

10 chuyến giả trong `FLIGHTS`:

| Tuyến | Chuyến | Giờ | Giá (VND) | Ghi chú |
|---|---|---|---|---|
| SGN→DAD 2026-10-07 | VN122 | 08:10 | 1.850.000 | hợp lệ |
| | VJ604 | 08:30 | 2.480.000 | quá đắt |
| | VJ606 | 10:30 | 1.350.000 | hợp lệ, rẻ nhất |
| | QH118 | 15:40 | 1.640.000 | chiều |
| SGN→HAN 2026-10-07 | VN210 | 06:30 | 2.300.000 | |
| | VJ150 | 09:45 | 1.450.000 | |
| | VN212 | 11:20 | 1.900.000 | |
| | QH220 | 13:00 | 1.100.000 | |
| HAN→SGN 2026-10-08 | VN301 | 07:00 | 1.700.000 | |
| | VJ302 | 18:30 | 1.200.000 | |

### 4.2 Tools

| Tool | Chức năng | Quyền |
|---|---|---|
| `search_flights(origin, destination, date)` | Tìm chuyến; `FAIL_SEARCH` giả lập timeout | allow |
| `book_seat(flight)` | Giữ chỗ, mã `<flight>-12A`, chưa thu tiền | allow, phải thỏa ràng buộc |
| `pay(code)` | Thanh toán, không hoàn tác | ask (cần người duyệt) |
| `get_booking(code)` | Đọc lại booking | allow |

Tool không có trong bảng quyền bị deny.

## 5. Implementation: Harness

| Yêu cầu đề | Implementation (`flight_agent.py`) | Nơi kiểm chứng |
|---|---|---|
| Constraint as data | `Constraints`, `to_prompt()`, `is_ok()` | `test_constraints_are_data`; case A–C |
| Completion by code | `is_done()` đọc lại booking | `test_done_is_checked_by_code`, `test_two_paid_bookings_is_not_done`; cột `got` |
| Permission | `PERMISSION`, `check()`, `APPROVE` | `test_booking_that_breaks_constraints_is_denied`, `test_payment_needs_human_approval`; case D1; cột `denied` |
| Handoff | `handoff()` | `test_handoff_reason_matches_failure`; case C, D, E2 |
| ReAct | `react()` | benchmark (mục 8) |
| Plan-then-Execute | `plan_execute()` | benchmark |
| Hybrid | `hybrid()` | benchmark |

Unit test: `pytest -q test_harness.py` (13 test, không cần LLM).

**Constraints là dữ liệu.** `Constraints(origin, destination, date, depart_before, max_price, note)`. `to_prompt()` sinh yêu cầu gửi model, `is_ok(flight)` kiểm tra bằng code. Prompt và kiểm tra cùng lấy từ một nguồn. Cần phân biệt hai loại thông tin:

- **Ràng buộc cứng có cấu trúc** (`origin`, `destination`, `date`, `depart_before`, `max_price`): harness bắt buộc kiểm qua `is_ok`.
- **Yêu cầu tự do** (`note`): văn bản người dùng, được đưa vào prompt để LLM đọc, nhưng harness không có schema để kiểm. `note` có thể mâu thuẫn với ràng buộc cứng (case F1).

**Hoàn thành bằng code.** `is_done()` đọc lại mọi booking. Done khi đúng một booking đã thanh toán và thỏa `is_ok`. Lời model không được tin. Unit test phủ: chưa book → false; book nhưng chưa pay → false; pay → true; hai booking đã trả tiền → false.

**Phân quyền.** `PERMISSION = {search_flights: allow, get_booking: allow, book_seat: allow, pay: ask}`; tool không liệt kê bị deny. `check()` chặn trước khi chạy: tool deny, call giống hệt đã chạy 3 lần (loop guard chặn lần thứ 4), `book_seat` vi phạm ràng buộc, `pay` chưa được duyệt. Call bị chặn trả về `status = denied` như dữ liệu, để model/planner thấy lý do. Kết quả `{}` là lỗi, không phải "không tìm thấy". Cả 3 pattern chỉ gọi tool qua `guarded()`; với `create_agent`, `guarded()` được gắn làm middleware `harness`.

**Handoff.** `handoff()` trả `reason` (ưu tiên `payment` → `tool_error` → `constraints`), `done_so_far`, `tried`, `question` để người tiếp quản biết đã làm gì và cần quyết định gì.

## 6. Ba pattern bắt buộc

Cùng model, tool, harness và `is_done()`. Khác nhau ở chỗ ai quyết định bước kế tiếp.

| | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| Ai chọn bước kế tiếp | LLM, mỗi bước | LLM viết plan một lần; code chạy | Code (flow cố định); LLM chỉ trong find |
| Số lần gọi LLM | biến thiên, tối đa 10 (`ModelCallLimit`) | đúng 1 | biến thiên, chỉ trong find: tối đa 2 lần invoke finder, mỗi lần tối đa 10 |
| Phục hồi khi tool lỗi | có (LLM tự retry) | không | có, chỉ ở bước find |

**ReAct**: `create_agent` + middleware `harness` + `ModelCallLimitMiddleware(10)`. Model quyết định từng bước dựa trên quan sát của bước trước; mỗi lần gọi gửi lại toàn bộ lịch sử.

**Plan-then-Execute**: code search trước (lỗi thì dừng, không gọi LLM); một lần `with_structured_output(Plan)` viết cả plan; code chạy từng bước, thay `$booking_code` bằng mã thật; bước nào lỗi thì dừng, không replan. Pattern cố ý không có vòng phản hồi, làm đối chứng với ReAct.

```mermaid
flowchart TD
    IN(["Constraints"]) --> A["search (code)"]
    A -->|"lỗi"| Z(["dừng"])
    A -->|"ok"| C["plan (1 lần LLM)"]
    C --> E["execute từng bước (code)<br/>guarded()"]
    E -->|"lỗi / denied"| Z
    E -->|"hết bước"| V{"is_done()"}
    V --> O(["DONE / FAILED + handoff"])
```

**Hybrid**: kết hợp *suy luận agentic* cho bước cần thích ứng với *thực thi tất định bằng code* cho các bước nhạy cảm. Find là bước không chắc chắn (tool có thể lỗi), nên giao cho một agent ReAct nhỏ chỉ có `search_flights` (read-only); agent tự retry trong một lần invoke, và code cho thêm tối đa 1 lần invoke lại nếu vẫn chưa có kết quả. Các bước có hậu quả (chọn chuyến, đặt, trả tiền, xác nhận) do code làm: select lọc `is_ok` và lấy chuyến rẻ nhất; book; pay (cần duyệt); verify bằng `is_done()`.

```mermaid
flowchart TD
    IN(["Constraints"]) --> F["find: ReAct, chỉ search_flights<br/>tối đa 2 lần invoke finder"]
    F --> SEL["select (code): is_ok, rẻ nhất"]
    SEL -->|"không có"| V
    SEL -->|"có"| BK["book_seat"] --> PY["pay (cần duyệt)"] --> V{"is_done()"}
    V --> O(["DONE / FAILED + handoff"])
```

## 7. Thiết lập thực nghiệm

**Case** (`evaluate.py`, 15 case, 6 nhóm):

| Nhóm | Case | Kỳ vọng |
|---|---|---|
| A. Đơn giản | A1–A3 | DONE |
| B. Nhiều ràng buộc | B1–B3 | DONE |
| C. Bất khả thi | C1–C4 | FAILED:constraints |
| D. Từ chối thanh toán | D1 | FAILED:payment |
| E. Lỗi tool | E1 (lỗi 1 lần), E2 (lỗi mãi) | DONE / FAILED:tool_error |
| F. Người dùng đẩy chuyến sai | F1 (muốn QH118), F2 (rẻ nhất, mọi giờ) | F1: FAILED:constraints; F2: DONE |

**Chỉ số.**

- **pass%** (case pass rate, cột `success%` trong output `evaluate.py`): tỉ lệ run có kết quả đúng kỳ vọng `DONE` / `FAILED:<reason>`. Đây không phải tỉ lệ đặt vé thành công: một case C kết thúc `FAILED:constraints` đúng lý do vẫn tính là pass.
- **handoff%**: pass% chỉ trên các case phải FAILED (C, D1, E2, F1).
- **tool_calls**, **denied** (call bị harness chặn), **tokens**, **latency**: trung bình mỗi run.

**Cấu hình.** `ag/gemini-3.8-flash-low` qua 9router, `temperature=0`. Chạy `python evaluate.py --repeats 3` (mặc định 3 pattern bắt buộc). Run tuần tự vì trạng thái là biến module.

**Lưu ý token.** Router cộng ~2000 token mỗi lần gọi LLM. Token dùng để so sánh giữa các pattern trên cùng hạ tầng, không phải số tuyệt đối của thuật toán. Latency dao động 5–75 s, nên ưu tiên tool_calls và kết quả.

## 8. Kết quả

### 8.1 Benchmark chính (15 case × 3 lần)

| Pattern | pass% | handoff% | tool_calls | denied | tokens | latency (s) |
|---|---|---|---|---|---|---|
| ReAct | **100.0** | 100.0 | 2.5 | 0.2 | 9291 | 31.5 |
| Plan-then-Execute | 93.3 | 100.0 | 2.5 | 0.1 | 2404 | 10.6 |
| Hybrid | 93.3 | 85.7 | 2.7 | 0.3 | 5797 | 30.2 |

pass% theo nhóm:

| Nhóm | ReAct | Plan-then-Execute | Hybrid |
|---|---|---|---|
| A–D | 100 | 100 | 100 |
| E. Lỗi tool | 100 | **50** | 100 |
| F. Đẩy chuyến sai | 100 | 100 | **50** |

- **Plan-then-Execute hỏng E1 (3/3)**: search lỗi một lần là dừng, không gọi model. Đây đúng là failure mode mà pattern không có vòng quan sát phải bộc lộ.
- **Hybrid hỏng F1 (3/3)**: người dùng muốn QH118, hệ thống đặt VJ606 và báo DONE mà không hỏi. Select là code, không đọc `note`.
- Denied trong benchmark đến từ D1 (pay bị từ chối) và loop guard ở E2. Không có `book_seat` vi phạm nào bị chặn, nên khả năng chặn đặt sai chỉ được chứng minh bằng unit test.

### 8.2 Đọc kết quả

ReAct đạt pass% cao nhất nhưng dùng nhiều token nhất. Plan-then-Execute dùng ít token hơn ReAct khoảng 3.9× (2.4k so với 9.3k), nhưng không phục hồi được lỗi tool. Hybrid dùng ít token hơn ReAct khoảng 1.6× và kiểm soát được flow, nhưng không nhận biết yêu cầu nằm trong `note`. Token ở đây chưa quy ra chi phí tiền. Với 3 lần lặp, 93.3% so với 100% là đúng 3 run hỏng: tín hiệu định hướng, không phải kết luận thống kê.

## 9. Thảo luận

**F1 và giới hạn của schema ràng buộc.** F1 (*"I love flight QH118, please book exactly that one."*) mâu thuẫn với `depart_before=12:00`. Yêu cầu "phải là QH118" chỉ nằm trong `note` tự do, chưa được nâng thành ràng buộc có cấu trúc. Harness vẫn bảo đảm mọi ràng buộc cứng đã khai báo: VJ606 thỏa tất cả, nên `is_ok` và `is_done()` đều cho qua. Nhưng Hybrid chọn chuyến bằng code, không đọc `note`, nên lặng lẽ thay bằng chuyến khác thay vì hỏi lại người dùng. ReAct và Plan-then-Execute qua được F1 vì LLM có đọc `note`. Hướng sửa là đưa yêu cầu ngữ nghĩa vào schema (ví dụ trường `flight` bắt buộc) để harness kiểm được; báo cáo này chưa thử hướng đó.

**Giới hạn.** 3 lần lặp, một model, dữ liệu mock 10 chuyến. Model hầu như không thử đặt chuyến vi phạm, nên khả năng chặn của harness chủ yếu được kiểm bằng unit test.

## 10. Kết luận

Trong ba pattern bắt buộc, ReAct đạt pass% cao nhất (100%) nhưng tốn token nhất (≈9.3k/run). Plan-then-Execute ít token nhất (≈2.4k) nhưng không phục hồi được lỗi tool (E1 hỏng 3/3). Hybrid kiểm soát flow và ít token hơn ReAct (≈5.8k), nhưng hỏng F1 (3/3) vì yêu cầu nằm trong `note` tự do, ngoài schema mà harness kiểm được.

Từ hạn chế của Hybrid, nhóm thử thêm cải tiến Hybrid + Replan ở Phụ lục A, đạt 100% trên benchmark. Kết luận chính vẫn là so sánh ba pattern bắt buộc; benchmark quy mô nhỏ nên chưa đủ để xếp hạng chắc chắn.

---

## Phụ lục A: Thí nghiệm mở rộng (Bonus)

Chạy: `python evaluate.py --patterns hybrid_replan --repeats 3`.

**A.1 Hybrid + Replan.** LLM lập plan một lần; code chạy và quan sát từng bước. Lỗi tạm thời được sửa tại chỗ bằng retry (tối đa 2 lần, không gọi LLM). Bước bị `denied`/`not_found` hoặc lỗi còn lại sau retry thì replan (tối đa 2 lần), planner thấy lý do lỗi trong `So far`. Không search lại khi replan. Đúng đắn ở F1 vẫn dựa vào planner LLM đọc `note`. Kiểm chứng đường đi bằng `test_replan_*` với planner giả: retry lỗi tool mà không replan; lỗi kéo dài → handoff; planner thấy lý do bị deny; không search lại; người từ chối thanh toán → không replan.

**A.2 Kết quả (15 case × 3 lần)**

| Pattern | pass% | handoff% | tool_calls | denied | tokens | latency (s) |
|---|---|---|---|---|---|---|
| Hybrid + Replan | 100.0 | 100.0 | 2.9 | 0.1 | 2483 | 5.8 |

Đạt 100% ở mọi nhóm, gồm F1.

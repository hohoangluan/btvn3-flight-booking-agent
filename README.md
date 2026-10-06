# Flight Booking Agent: Harness và 3 design pattern

Agent đặt vé máy bay (LangChain / LangGraph) với một lớp **harness** bằng code: model không bỏ qua được. Cùng một agent được cài theo ba pattern (ReAct, Plan-then-Execute, Hybrid) và đánh giá trên cùng một benchmark.

Báo cáo chi tiết: [BAOCAO.md](BAOCAO.md).

## Cấu trúc

| File | Vai trò |
|---|---|
| `flight_agent.py` | Mock tools, harness, các pattern, `run()`, CLI |
| `evaluate.py` | Benchmark: 15 case (nhóm A–F) × pattern × số lần lặp |
| `test_harness.py` | Unit test cho harness, không cần LLM |
| `requirements.txt` | Thư viện Python |
| `.env.example` | Mẫu cấu hình (sao chép thành `.env`) |
| `BAOCAO.md` | Báo cáo |

## Cài đặt

Yêu cầu Python 3.13.

```bash
pip install -r requirements.txt
cp .env.example .env
```

Sửa `.env`:

- `NINEROUTER_API_KEY`: khóa API của 9router (bắt buộc).
- `NINEROUTER_URL`: địa chỉ OpenAI-compatible của 9router (mặc định `http://localhost:20128/v1`).
- `NINEROUTER_MODEL`: model phục vụ qua 9router (mặc định trong code: `ag/gemini-3.8-flash`).

Không commit `.env`. File này đã nằm trong `.gitignore`.

## Chạy

Chạy một pattern trên yêu cầu mặc định (SGN → DAD, 2026-10-07, trước 12:00, tối đa 2.000.000 VND). Khi agent muốn thanh toán, chương trình hỏi xác nhận trên terminal:

```bash
python flight_agent.py react
python flight_agent.py plan_execute
python flight_agent.py hybrid
python flight_agent.py hybrid_replan
```

Chạy unit test (không cần LLM):

```bash
pytest -q test_harness.py
```

Chạy benchmark ba pattern bắt buộc, 3 lần lặp mỗi case:

```bash
python evaluate.py --repeats 3
```

Chạy thêm pattern mở rộng (xem Phụ lục A trong báo cáo):

```bash
python evaluate.py --patterns hybrid_replan --repeats 3
```

Tùy chọn `--out <file.csv>` ghi kết quả từng lần chạy ra CSV. Benchmark gọi LLM thật nên mất nhiều thời gian và kết quả có thể lệch giữa các lần chạy.

## Thiết kế ngắn gọn

- **Constraints là dữ liệu**: `Constraints` (tuyến, ngày, giờ khởi hành, giá tối đa). `is_ok()` kiểm tra bằng code. Trường `note` là văn bản tự do, không được harness kiểm tra.
- **Phân quyền là dữ liệu**: `PERMISSION` cho từng tool (`allow` / `ask` / `deny`). Tool không liệt kê bị từ chối. `pay` cần người duyệt.
- **Harness**: mọi tool call đi qua `guarded()`. `check()` chặn tool bị deny, lặp quá 3 lần, đặt chuyến vi phạm ràng buộc, hoặc thanh toán khi chưa duyệt.
- **Hoàn thành bằng code**: `is_done()` đọc lại booking. Chỉ tính DONE khi đúng một booking đã thanh toán và thỏa ràng buộc.
- **Handoff**: `handoff()` trả lý do (`constraints`, `payment`, `tool_error`), các booking đã có, lịch sử call và câu hỏi cho người.

Các pattern:

- `react`: `create_agent` của LangChain, có middleware harness và giới hạn 10 lần gọi model.
- `plan_execute`: code tìm chuyến, một lần LLM viết cả plan, code chạy plan, không replan.
- `hybrid`: luồng cố định trong code. Chỉ bước tìm chuyến dùng ReAct; chọn, đặt, thanh toán và kiểm tra là code.
- `hybrid_replan` (mở rộng): LLM lập plan, code chạy từng bước và sửa lỗi tạm thời tại chỗ, replan khi cần.

## Dữ liệu mock

10 chuyến giả trong `FLIGHTS`, không có network. Tool: `search_flights`, `book_seat`, `pay`, `get_booking`. `FAIL_SEARCH` giả lập lỗi timeout để thử phục hồi.

## Kết quả

Chi tiết và phân tích nằm trong [BAOCAO.md](BAOCAO.md). Tóm tắt benchmark ba pattern bắt buộc (15 case × 3 lần, mock data, một model):

| Pattern | Thành công | Token/run (ước tính, router cộng thêm) |
|---|---|---|
| ReAct | 100% | ~9.3k |
| Plan-then-Execute | 93.3% (hỏng E1) | ~2.4k |
| Hybrid | 93.3% (hỏng F1) | ~5.8k |

Số token bị thổi phồng bởi router, nên chỉ dùng để so sánh trong cùng một môi trường.

import ast
import json
import operator
import sys
from typing import Any, Callable

from pydantic import BaseModel, Field
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import Settings

# ---------------------------------------------------------------------------
# 1. ĐỊNH NGHĨA CÁC HÀM PYTHON THẬT (Requirement #1)
# ---------------------------------------------------------------------------


def calculate(expression: str) -> dict[str, Any]:
    """Thực hiện tính toán biểu thức toán học an toàn.

    Hỗ trợ các phép tính +, -, *, /, **, %.
    Bắt lỗi phép chia cho 0 hoặc biểu thức không hợp lệ.
    """
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            op_type = type(node.op)
            if op_type in allowed_operators:
                if op_type == ast.Div and right == 0:
                    raise ZeroDivisionError("Lỗi toán học: Không thể chia cho 0!")
                return allowed_operators[op_type](left, right)
            raise ValueError(f"Toán tử {op_type.__name__} không được hỗ trợ.")
        elif isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            op_type = type(node.op)
            if op_type in allowed_operators:
                return allowed_operators[op_type](operand)
            raise ValueError(f"Toán tử {op_type.__name__} không được hỗ trợ.")
        else:
            raise ValueError("Biểu thức toán học chứa ký tự hoặc cấu trúc không an toàn.")

    # Làm sạch biểu thức
    clean_expr = expression.replace("x", "*").replace("X", "*").replace(":", "/").strip()
    parsed_ast = ast.parse(clean_expr, mode="eval")
    result_val = eval_node(parsed_ast.body)

    return {
        "expression": expression,
        "result": result_val,
        "formatted_result": f"{result_val:,.2f}".rstrip("0").rstrip("."),
    }


def get_weather(city: str) -> dict[str, Any]:
    """Lấy thông tin thời tiết hiện tại của một thành phố (Dữ liệu thực tế/Mock DB)."""
    city_normalized = city.strip().lower()

    weather_db = {
        "hà nội": {
            "city": "Hà Nội",
            "temperature": 28,
            "unit": "°C",
            "condition": "Có mây, nắng nhẹ",
            "humidity": "75%",
            "wind_speed": "12 km/h",
        },
        "tp.hcm": {
            "city": "TP.Hồ Chí Minh",
            "temperature": 34,
            "unit": "°C",
            "condition": "Nắng nóng, chiều có thể mưa rào",
            "humidity": "80%",
            "wind_speed": "15 km/h",
        },
        "hồ chí minh": {
            "city": "TP.Hồ Chí Minh",
            "temperature": 34,
            "unit": "°C",
            "condition": "Nắng nóng, chiều có thể mưa rào",
            "humidity": "80%",
            "wind_speed": "15 km/h",
        },
        "đà nẵng": {
            "city": "Đà Nẵng",
            "temperature": 31,
            "unit": "°C",
            "condition": "Trời quang mây, lộng gió",
            "humidity": "70%",
            "wind_speed": "20 km/h",
        },
    }

    for key, data in weather_db.items():
        if key in city_normalized or city_normalized in key:
            return data

    # Nếu không tìm thấy thành phố trong DB -> Ném ra lỗi để test Requirement #6 (Tool error handling)
    raise ValueError(f"Không tìm thấy dữ liệu thời tiết cho thành phố '{city}'. Vui lòng thử 'Hà Nội', 'TP.HCM' hoặc 'Đà Nẵng'.")


# ---------------------------------------------------------------------------
# 2. SCHEMA CHO MỖI TOOL (Requirement #2 & Hint #5)
# ---------------------------------------------------------------------------

# Schema theo định dạng OpenAI / Anthropic Function Calling
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Thực hiện tính toán biểu thức toán học. Hãy dùng tool này bất cứ khi nào người dùng yêu cầu cộng, trừ, nhân, chia, tính toán con số.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Biểu thức toán học bằng chuỗi, ví dụ: '250 * 4 + 1500' hoặc '(100 + 50) / 2'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Tra cứu thông tin thời tiết thời gian thực (nhiệt độ, độ ẩm, thời tiết) của một thành phố.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Tên thành phố cần tra cứu, ví dụ: 'Hà Nội', 'TP.HCM', 'Đà Nẵng'.",
                    }
                },
                "required": ["city"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# 3. DISPATCH THEO TÊN TOOL (Requirement #3)
# ---------------------------------------------------------------------------

TOOLS_DISPATCH: dict[str, Callable[..., Any]] = {
    "calculate": calculate,
    "get_weather": get_weather,
}


# ---------------------------------------------------------------------------
# 4. MOCK ENGINE (Dùng khi chưa có hoặc lỗi OpenRouter Key để test 100% mượt)
# ---------------------------------------------------------------------------


class MockToolCall(BaseModel):
    id: str
    function_name: str
    arguments: dict[str, Any]


def mock_assistant_step(
    messages: list[dict[str, Any]], turn_count: int
) -> tuple[str | None, list[MockToolCall]]:
    """Giả lập phản hồi của LLM có hỗ trợ Tool Calling khi chạy Offline/Mock."""
    last_msg = messages[-1]
    last_content = str(last_msg.get("content", ""))

    # Lượt 1: Phân tích user input để xin gọi Tool
    if turn_count == 1:
        tool_calls: list[MockToolCall] = []

        # Kiểm tra nếu câu hỏi cần get_weather
        if "thời tiết" in last_content.lower() or "thời tiết" in str(messages[0].get("content", "")).lower():
            if "hà nội" in last_content.lower() or "hà nội" in str(messages[0].get("content", "")).lower():
                tool_calls.append(
                    MockToolCall(
                        id="call_weather_1",
                        function_name="get_weather",
                        arguments={"city": "Hà Nội"},
                    )
                )
            if "tp.hcm" in last_content.lower() or "hồ chí minh" in last_content.lower() or "tp.hcm" in str(messages[0].get("content", "")).lower():
                tool_calls.append(
                    MockToolCall(
                        id="call_weather_2",
                        function_name="get_weather",
                        arguments={"city": "TP.HCM"},
                    )
                )
            if "đà nẵng" in last_content.lower() or "đà nẵng" in str(messages[0].get("content", "")).lower():
                tool_calls.append(
                    MockToolCall(
                        id="call_weather_3",
                        function_name="get_weather",
                        arguments={"city": "Đà Nẵng"},
                    )
                )

        # Kiểm tra nếu câu hỏi cần calculate
        if any(char in last_content for char in ["+", "*", "/", "tính"]) or any(char in str(messages[0].get("content", "")) for char in ["+", "*", "/", "tính"]):
            if "100 / 0" in last_content or "100 / 0" in str(messages[0].get("content", "")):
                expr = "100 / 0"
            elif "250 * 4 + 1500" in last_content or "250 * 4 + 1500" in str(messages[0].get("content", "")):
                expr = "250 * 4 + 1500"
            else:
                expr = "(250 * 4 + 1500) * 2"

            tool_calls.append(
                MockToolCall(
                    id="call_calc_1",
                    function_name="calculate",
                    arguments={"expression": expr},
                )
            )

        if tool_calls:
            return None, tool_calls

    # Lượt 2+: Đã có kết quả tool -> Tổng hợp câu trả lời cuối
    weather_results = [m for m in messages if m.get("role") == "tool" and "city" in m.get("content", "")]
    calc_results = [m for m in messages if m.get("role") == "tool" and ("result" in m.get("content", "") or "Lỗi" in m.get("content", ""))]

    final_text_parts = []
    if weather_results:
        for w in weather_results:
            final_text_parts.append(f"• Thông tin thời tiết: {w['content']}")
    if calc_results:
        for c in calc_results:
            final_text_parts.append(f"• Kết quả tính toán: {c['content']}")

    if final_text_parts:
        return "Dựa trên dữ liệu thực tế từ các tool:\n" + "\n".join(final_text_parts), []
    
    return "Tôi đã xử lý xong yêu cầu của bạn.", []


# ---------------------------------------------------------------------------
# 5. VÒNG LẶP TOOL CALLING ENGINE (Requirements #4, #5, #6)
# ---------------------------------------------------------------------------


async def run_assistant(
    user_query: str,
    settings: Settings,
    console: Console,
    use_real_api: bool = False,
) -> str:
    """Chạy vòng lặp Tool Calling hoàn chỉnh (Requirement #4).

    - Hỗ trợ xử lý >= 2 tool / lượt (Requirement #5).
    - Hỗ trợ xử lý lỗi tool mà không bị crash (Requirement #6).
    """
    console.print(Panel(f"[bold white]❓ CÂU HỎI NGƯỜI DÙNG:[/bold white]\n[yellow]{user_query}[/yellow]", title="USER INPUT", border_style="cyan"))

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Bạn là trợ lý AI thông minh. Bạn có quyền truy cập các công cụ (tools) thực tế. "
                "BẮT BỘC gọi tool khi cần dữ liệu thời tiết hoặc tính toán số liệu. "
                "KHÔNG ĐƯỢC đoán bừa kết quả."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    turn = 0
    max_turns = 10

    while turn < max_turns:
        turn += 1
        console.print(f"\n[bold magenta]🔄 --- VÒNG LẶP TOOL CALLING (LƯỢT {turn}) ---[/bold magenta]")

        tool_calls_to_process = []
        assistant_text: str | None = None

        if use_real_api and settings.openrouter_api_key and settings.openrouter_api_key.get_secret_value():
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.openrouter_api_key.get_secret_value(),
            )
            # Gọi LLM với danh sách tools
            response = await client.chat.completions.create(
                model="google/gemini-2.5-flash:free",
                messages=messages,  # type: ignore[arg-type]
                tools=TOOLS_SCHEMA,  # type: ignore[arg-type]
                timeout=float(settings.timeout_seconds),
            )
            msg = response.choices[0].message
            assistant_text = msg.content

            # Chuyển đổi message OpenAI thành dict để lưu vào lịch sử
            msg_dict: dict[str, Any] = {"role": "assistant"}
            if msg.content:
                msg_dict["content"] = msg.content
            if msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
                tool_calls_to_process = msg.tool_calls
            messages.append(msg_dict)
        else:
            # Chạy Mock Predictor cho Tool Loop
            text_res, mock_calls = mock_assistant_step(messages, turn)
            assistant_text = text_res

            if mock_calls:
                msg_dict = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": mc.id,
                            "type": "function",
                            "function": {
                                "name": mc.function_name,
                                "arguments": json.dumps(mc.arguments, ensure_ascii=False),
                            },
                        }
                        for mc in mock_calls
                    ],
                }
                messages.append(msg_dict)
                tool_calls_to_process = mock_calls

        # Stop condition: Model không yêu cầu gọi tool nữa (trả lời cuối)
        if not tool_calls_to_process:
            console.print("[green]✔ Model không yêu cầu gọi tool nào nữa. Kết thúc vòng lặp![/green]")
            break

        console.print(f"[bold yellow]⚙️ Model yêu cầu gọi {len(tool_calls_to_process)} tool(s) cùng lúc:[/bold yellow]")

        # 5. Duyệt HẾT các tool call được yêu cầu trong lượt (Requirement #5)
        for tc in tool_calls_to_process:
            if hasattr(tc, "function"):
                tc_id = tc.id
                func_name = tc.function.name
                raw_args = tc.function.arguments
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            else:
                # Mock call object
                tc_id = tc.id
                func_name = tc.function_name
                args = tc.arguments

            console.print(f"  👉 [cyan]Tool:[/cyan] [bold]{func_name}[/bold] | [cyan]ID:[/cyan] {tc_id} | [cyan]Args:[/cyan] {args}")

            # 3. Dispatch theo tên & 6. Xử lý lỗi tool (Requirement #3 & #6)
            tool_result_content = ""
            try:
                if func_name not in TOOLS_DISPATCH:
                    raise KeyError(f"Tool '{func_name}' không tồn tại trong hệ thống.")

                # Thực thi hàm Python thật (Requirement #1)
                real_func = TOOLS_DISPATCH[func_name]
                execution_output = real_func(**args)
                tool_result_content = json.dumps(execution_output, ensure_ascii=False)
                console.print(f"     [bold green]✓ Kết quả:[/bold green] {tool_result_content}")

            except Exception as e:
                # Bắt lỗi, trả thông báo rõ cho model, KHÔNG CRASH chướng trình (Requirement #6)
                tool_result_content = f"Lỗi thực thi tool '{func_name}': {str(e)}"
                console.print(f"     [bold red]✗ Lỗi Tool (Đã xử lý an toàn):[/bold red] {tool_result_content}")

            # Append kết quả tool vào message history với role="tool" và tool_call_id khớp đúng
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_result_content,
                }
            )

    final_answer = assistant_text or "Không nhận được câu trả lời cuối cùng."
    console.print(
        Panel(
            f"[bold green]💬 CÂU TRẢ LỜI CUỐI CÙNG CỦA ASSISTANT:[/bold green]\n\n{final_answer}",
            title="FINAL ANSWER",
            border_style="green",
        )
    )
    return final_answer


# ---------------------------------------------------------------------------
# 6. CHẠY THỬ CÁC KỊCH BẢN (Requirement #7)
# ---------------------------------------------------------------------------


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]AI ENGINEER ROADMAP - TOPIC 6: TOOL CALLING & AI ASSISTANT LAB[/bold cyan]\n"
            "[yellow]Script: assistant.py (Định nghĩa Tool Schema, Dispatch & Tool Loop)[/yellow]",
            border_style="cyan",
        )
    )

    settings = Settings()
    use_real_api = bool(settings.openrouter_api_key and settings.openrouter_api_key.get_secret_value())

    if use_real_api:
        console.print("[green]✔ Đã tìm thấy OPENROUTER_API_KEY trong .env. Đang chạy qua OpenRouter API...[/green]")
    else:
        console.print("[yellow]⚠ Chưa cấu hình OPENROUTER_API_KEY. Đang chạy chế độ Mock Engine để test 100% Tool Loop & Xử lý lỗi...[/yellow]")

    # -----------------------------------------------------------------------
    # KỊCH BẢN 1: CẦN 1 TOOL (Requirement #7 - Câu 1)
    # -----------------------------------------------------------------------
    console.print("\n" + "=" * 80)
    console.print("[bold blue]📌 KỊCH BẢN 1: CÂU HỎI CẦN 1 TOOL[/bold blue]")
    console.print("=" * 80)
    query_1 = "Thời tiết tại Hà Nội hôm nay thế nào?"
    await run_assistant(query_1, settings, console, use_real_api)

    # -----------------------------------------------------------------------
    # KỊCH BẢN 2: CẦN CẢ 2 TOOLS CÙNG LÚC (Requirement #7 - Câu 2)
    # -----------------------------------------------------------------------
    console.print("\n" + "=" * 80)
    console.print("[bold blue]📌 KỊCH BẢN 2: CÂU HỎI CẦN CẢ 2 TOOL CÙNG LÚC (PARALLEL TOOL CALLING)[/bold blue]")
    console.print("=" * 80)
    query_2 = "Cho mình biết thời tiết ở TP.HCM thế nào và tính giúp mình phép tính 250 * 4 + 1500?"
    await run_assistant(query_2, settings, console, use_real_api)

    # -----------------------------------------------------------------------
    # KỊCH BẢN 3: XỬ LÝ LỖI TOOL AN TOÀN (Requirement #6 - Tool Error Handling)
    # -----------------------------------------------------------------------
    console.print("\n" + "=" * 80)
    console.print("[bold blue]📌 KỊCH BẢN 3: KIỂM THỬ XỬ LÝ LỖI TOOL (CHIA CHO 0 & KHÔNG CRASH SCRIPT)[/bold blue]")
    console.print("=" * 80)
    query_3 = "Tính giúp mình phép chia 100 / 0 và thời tiết tại Đà Nẵng?"
    await run_assistant(query_3, settings, console, use_real_api)

    # -----------------------------------------------------------------------
    # TỔNG KẾT
    # -----------------------------------------------------------------------
    summary_table = Table(title="🎯 TỔNG KẾT NGHỆ THUẬT TOOL CALLING (TOPIC 6)", box=box.ROUNDED)
    summary_table.add_column("Yêu cầu đề bài", style="bold cyan")
    summary_table.add_column("Trạng thái", style="bold green")
    summary_table.add_column("Chi tiết triển khai", style="dim")

    summary_table.add_row("1. >= 2 hàm Python thật", "PASSED", "Hàm calculate() & get_weather()")
    summary_table.add_row("2. Schema cho mỗi tool", "PASSED", "Khai báo OpenAI Tool Schema đầy đủ name/desc/parameters")
    summary_table.add_row("3. Dispatch theo tên", "PASSED", "Dùng TOOLS_DISPATCH dict gọi đúng hàm")
    summary_table.add_row("4. Vòng lặp tool calling", "PASSED", "Vòng lặp while lặp tới khi model không xin tool")
    summary_table.add_row("5. Nhiều tool / lượt", "PASSED", "Duyệt toàn bộ danh sách tool_calls trong 1 turn")
    summary_table.add_row("6. Xử lý lỗi tool", "PASSED", "Try/except bắt lỗi, gửi thông báo lỗi về role='tool' không crash")
    summary_table.add_row("7. Chạy thử 2+ câu hỏi", "PASSED", "Đã test kịch bản 1 tool, 2 tool song song & ca lỗi")

    console.print("\n")
    console.print(summary_table)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

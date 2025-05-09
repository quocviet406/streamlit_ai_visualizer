# prompts.py

def build_prompt(user_prompt, df_head):
    return f"""
Bạn là một chuyên gia phân tích dữ liệu. Dưới đây là yêu cầu từ người dùng:

Yêu cầu: "{user_prompt}"

Dữ liệu mẫu như sau (5 dòng đầu):
{df_head}

Hãy viết mã Python (sử dụng pandas và matplotlib hoặc plotly) để phân tích và trực quan hóa theo yêu cầu trên.


Với các chart thêm lable, title, legend, giá trị trả ve và các thông số khác để biểu đồ dễ hiểu hơn.
Dữ liệu đầu vào là một DataFrame pandas có tên là df.

Hãy bỏ đoạn diễn giải và chỉ trả về mã Python mà không có bất kỳ chú thích nào.
"""

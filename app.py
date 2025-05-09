import streamlit as st
import pandas as pd
from openai import OpenAI
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import re
import json
import io
import base64
from dotenv import load_dotenv
from prompts import build_prompt

# Thêm thư viện để kết nối SQL Server
try:
    import pyodbc
    import sqlalchemy
    SQL_AVAILABLE = True
except ImportError:
    SQL_AVAILABLE = False

# Load biến môi trường từ file .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Khởi tạo OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="AI Trợ Lý Phân Tích Dữ Liệu", layout="wide")
st.title("📊 AI Trợ Lý Phân Tích Dữ Liệu Tự Động")

# Khởi tạo session state để lưu lịch sử hội thoại
if "messages" not in st.session_state:
    st.session_state.messages = []

if "df" not in st.session_state:
    st.session_state.df = None

if "data_source" not in st.session_state:
    st.session_state.data_source = "file"

if "sql_tables" not in st.session_state:
    st.session_state.sql_tables = []

if "sql_connection" not in st.session_state:
    st.session_state.sql_connection = {
        "server": "localhost",
        "database": "master",
        "username": None,
        "password": None,
        "use_windows_auth": False
    }

if "visualization_tool" not in st.session_state:
    st.session_state.visualization_tool = "plotly"

# Hàm để lưu biểu đồ matplotlib thành base64
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

# Hàm để lưu biểu đồ plotly thành JSON
def plotly_fig_to_json(fig):
    return fig.to_json()

# Hàm để xuất dữ liệu cho Power BI (sử dụng CSV)
def export_for_powerbi(df):
    # Tạo file CSV trong bộ nhớ
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()

# Hàm kết nối SQL Server và lấy dữ liệu
def connect_to_sql_server(server, database, username=None, password=None, query=None, use_windows_auth=False):
    try:
        # Tạo chuỗi kết nối SQLAlchemy
        if use_windows_auth:
            connection_uri = f"mssql+pyodbc://{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
        else:
            connection_uri = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
        
        # Tạo engine SQLAlchemy
        engine = sqlalchemy.create_engine(connection_uri)
        
        # Thực thi truy vấn nếu có
        if query:
            df = pd.read_sql(query, engine)
            return df, None
        else:
            return engine, None
    except Exception as e:
        return None, str(e)

# Hàm để lấy danh sách bảng từ SQL Server
def get_sql_tables(server, database, username=None, password=None, use_windows_auth=False):
    try:
        # Sử dụng hàm kết nối đã có
        engine, error = connect_to_sql_server(server, database, username, password, None, use_windows_auth)
        if error:
            return [], error
        
        # Truy vấn để lấy danh sách bảng
        query = """
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
        """
        
        # Thực thi truy vấn
        tables_df = pd.read_sql(query, engine)
        
        return tables_df['TABLE_NAME'].tolist(), None
    except Exception as e:
        return [], str(e)

# Hàm để lấy cấu trúc bảng từ SQL Server
def get_table_structure(server, database, username=None, password=None, table_name=None, use_windows_auth=False):
    try:
        # Sử dụng hàm kết nối đã có
        engine, error = connect_to_sql_server(server, database, username, password, None, use_windows_auth)
        if error:
            return None, error
        
        # Truy vấn để lấy cấu trúc bảng
        query = f"""
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table_name}'
        ORDER BY ORDINAL_POSITION
        """
        
        # Thực thi truy vấn
        columns_df = pd.read_sql(query, engine)
        
        return columns_df, None
    except Exception as e:
        return None, str(e)

# Sidebar cho tải dữ liệu và cài đặt
with st.sidebar:
    st.header("📁 Dữ liệu & Cài đặt")
    
    # Chọn nguồn dữ liệu
    st.session_state.data_source = st.radio(
        "Chọn nguồn dữ liệu",
        ["File", "SQL Server"],
        index=0 if st.session_state.data_source == "file" else 1
    ).lower()
    
    if st.session_state.data_source == "file":
        # Tải dữ liệu từ file
        uploaded_file = st.file_uploader("Tải lên file dữ liệu (.csv hoặc .xlsx)", type=["csv", "xlsx"])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                st.success(f"✅ Đã tải dữ liệu: {uploaded_file.name}")
                
                # Hiển thị thông tin dữ liệu
                with st.expander("📊 Thông tin dữ liệu"):
                    st.write(f"Số dòng: {st.session_state.df.shape[0]}")
                    st.write(f"Số cột: {st.session_state.df.shape[1]}")
                    st.write("Các cột:")
                    st.write(", ".join(st.session_state.df.columns.tolist()))
                    
                # Tùy chọn xuất dữ liệu cho Power BI
                if st.session_state.df is not None:
                    st.download_button(
                        label="📥 Xuất dữ liệu cho Power BI (CSV)",
                        data=export_for_powerbi(st.session_state.df),
                        file_name="data_for_powerbi.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"❌ Lỗi khi tải dữ liệu: {e}")
    
    else:  # SQL Server
        if not SQL_AVAILABLE:
            st.error("⚠️ Thư viện cần thiết chưa được cài đặt. Vui lòng cài đặt bằng lệnh: pip install pyodbc sqlalchemy")
        else:
            # Form kết nối SQL Server
            with st.form("sql_connection_form"):
                st.subheader("Kết nối SQL Server")
                
                # Thông tin kết nối
                server = st.text_input("Server", value="localhost")
                database = st.text_input("Database", value="master")
                
                # Chọn phương thức xác thực
                auth_method = st.radio("Phương thức xác thực", ["SQL Server Authentication", "Windows Authentication"])
                use_windows_auth = auth_method == "Windows Authentication"
                
                # Hiển thị trường username/password nếu dùng SQL Server Authentication
                username = None
                password = None
                if not use_windows_auth:
                    username = st.text_input("Username", value="sa")
                    password = st.text_input("Password", type="password")
                
                # Nút kết nối
                submit_button = st.form_submit_button("Kết nối")
                
                if submit_button:
                    tables, error = get_sql_tables(server, database, username, password, use_windows_auth)
                    if error:
                        st.error(f"❌ Lỗi kết nối: {error}")
                    else:
                        st.session_state.sql_tables = tables
                        st.success(f"✅ Đã kết nối thành công! Tìm thấy {len(tables)} bảng.")
                        # Lưu thông tin kết nối vào session state để sử dụng sau này
                        st.session_state.sql_connection = {
                            "server": server,
                            "database": database,
                            "username": username,
                            "password": password,
                            "use_windows_auth": use_windows_auth
                        }
            
            # Nếu đã kết nối thành công
            if st.session_state.sql_tables:
                # Chọn bảng hoặc nhập truy vấn SQL
                query_type = st.radio("Chọn cách lấy dữ liệu", ["Chọn bảng", "Nhập truy vấn SQL"])
                
                if query_type == "Chọn bảng":
                    selected_table = st.selectbox("Chọn bảng", st.session_state.sql_tables)
                    
                    if selected_table:
                        # Lấy thông tin kết nối từ session state
                        conn_info = st.session_state.sql_connection
                        
                        # Hiển thị cấu trúc bảng
                        columns_df, error = get_table_structure(
                            conn_info["server"], 
                            conn_info["database"], 
                            conn_info["username"], 
                            conn_info["password"], 
                            selected_table,
                            conn_info["use_windows_auth"]
                        )
                        if error:
                            st.error(f"❌ Lỗi khi lấy cấu trúc bảng: {error}")
                        else:
                            with st.expander("Cấu trúc bảng"):
                                st.dataframe(columns_df)
                        
                        # Tạo truy vấn SQL từ bảng đã chọn
                        query = f"SELECT * FROM [{selected_table}]"
                        
                        # Nút để lấy dữ liệu
                        if st.button("Lấy dữ liệu từ bảng"):
                            df, error = connect_to_sql_server(
                                conn_info["server"], 
                                conn_info["database"], 
                                conn_info["username"], 
                                conn_info["password"], 
                                query,
                                conn_info["use_windows_auth"]
                            )
                            if error:
                                st.error(f"❌ Lỗi khi truy vấn dữ liệu: {error}")
                            else:
                                st.session_state.df = df
                                st.success(f"✅ Đã lấy {df.shape[0]} dòng dữ liệu từ bảng {selected_table}")
                
                else:  # Nhập truy vấn SQL
                    query = st.text_area("Nhập truy vấn SQL", height=150, 
                                        value="SELECT TOP 1000 * FROM YourTableName")
                    
                    # Nút để thực thi truy vấn
                    if st.button("Thực thi truy vấn"):
                        # Lấy thông tin kết nối từ session state
                        conn_info = st.session_state.sql_connection
                        
                        df, error = connect_to_sql_server(
                            conn_info["server"], 
                            conn_info["database"], 
                            conn_info["username"], 
                            conn_info["password"], 
                            query,
                            conn_info["use_windows_auth"]
                        )
                        if error:
                            st.error(f"❌ Lỗi khi thực thi truy vấn: {error}")
                        else:
                            st.session_state.df = df
                            st.success(f"✅ Đã lấy {df.shape[0]} dòng dữ liệu từ truy vấn")
                
                # Hiển thị thông tin dữ liệu nếu có
                if st.session_state.df is not None:
                    with st.expander("📊 Thông tin dữ liệu"):
                        st.write(f"Số dòng: {st.session_state.df.shape[0]}")
                        st.write(f"Số cột: {st.session_state.df.shape[1]}")
                        st.write("Các cột:")
                        st.write(", ".join(st.session_state.df.columns.tolist()))
                    
                    # Tùy chọn xuất dữ liệu cho Power BI
                    st.download_button(
                        label="📥 Xuất dữ liệu cho Power BI (CSV)",
                        data=export_for_powerbi(st.session_state.df),
                        file_name="data_for_powerbi.csv",
                        mime="text/csv"
                    )
    
    # Chọn công cụ trực quan hóa
    st.session_state.visualization_tool = st.radio(
        "Công cụ trực quan hóa",
        ["Plotly/Matplotlib", "Power BI"]
    )
    
    # Cài đặt model
    model = st.selectbox(
        "Chọn model AI",
        ["gpt-4", "gpt-3.5-turbo"],
        index=0
    )
    
    # Nút xóa lịch sử
    if st.button("🗑️ Xóa lịch sử hội thoại"):
        st.session_state.messages = []
        st.success("Đã xóa lịch sử hội thoại")

# Hiển thị dữ liệu nếu có
if st.session_state.df is not None:
    with st.expander("Xem dữ liệu"):
        st.dataframe(st.session_state.df.head(10))

# Hiển thị hướng dẫn Power BI nếu được chọn
if st.session_state.visualization_tool == "Power BI":
    with st.expander("🔍 Hướng dẫn sử dụng Power BI", expanded=True):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ### Cách sử dụng dữ liệu với Power BI:
            
            1. Nhấn nút **📥 Xuất dữ liệu cho Power BI (CSV)** ở sidebar để tải xuống file CSV
            2. Mở Power BI Desktop trên máy tính của bạn
            3. Chọn **Get Data** > **Text/CSV** và chọn file vừa tải xuống
            4. Chọn các tùy chọn phù hợp và nhấn **Load**
            5. Bắt đầu tạo các biểu đồ trong Power BI
            """)
        
        with col2:
            if st.session_state.df is not None:
                st.markdown("### Ví dụ về mã DAX cho Power BI:")
                # Tạo ví dụ DAX
                columns = st.session_state.df.columns.tolist()
                numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
                date_cols = [col for col in columns if 'date' in col.lower() or 'time' in col.lower()]
                
                dax_examples = []
                
                # Tạo ví dụ về tính tổng
                if numeric_cols:
                    dax_examples.append(f"Tổng {numeric_cols[0]} = SUM(Data[{numeric_cols[0]}])")
                
                # Tạo ví dụ về tính trung bình
                if numeric_cols:
                    dax_examples.append(f"Trung bình {numeric_cols[0]} = AVERAGE(Data[{numeric_cols[0]}])")
                
                # Tạo ví dụ về đếm
                if columns:
                    dax_examples.append(f"Số lượng = COUNTROWS(Data)")
                
                # Tạo ví dụ về lọc theo thời gian
                if date_cols and numeric_cols:
                    dax_examples.append(f"""Tổng {numeric_cols[0]} năm hiện tại = 
CALCULATE(
    SUM(Data[{numeric_cols[0]}]),
    YEAR(Data[{date_cols[0]}]) = YEAR(TODAY())
)""")
                
                for example in dax_examples:
                    st.code(example, language="dax")

# Hiển thị lịch sử hội thoại
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Hiển thị biểu đồ nếu có
        if message["role"] == "assistant" and "chart_type" in message:
            if message["chart_type"] == "plotly":
                # Khôi phục biểu đồ plotly từ JSON
                fig_json = message["chart_data"]
                fig_dict = json.loads(fig_json)
                fig = go.Figure(fig_dict)
                st.plotly_chart(fig, use_container_width=True)
            
            elif message["chart_type"] == "matplotlib":
                # Hiển thị biểu đồ matplotlib từ base64
                img_str = message["chart_data"]
                st.image(f"data:image/png;base64,{img_str}", use_container_width=True)
            
            elif message["chart_type"] == "powerbi":
                # Hiển thị hướng dẫn Power BI
                st.info("💡 Dữ liệu đã được chuẩn bị cho Power BI. Tải xuống từ sidebar và làm theo hướng dẫn.")

# Nhập câu hỏi mới
if prompt := st.chat_input("Nhập câu hỏi hoặc yêu cầu phân tích..."):
    # Kiểm tra xem đã tải dữ liệu chưa
    if st.session_state.df is None:
        st.error("⚠️ Vui lòng tải dữ liệu trước khi đặt câu hỏi!")
    else:
        # Hiển thị câu hỏi của người dùng
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Lưu câu hỏi vào lịch sử
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Xử lý câu hỏi
        with st.chat_message("assistant"):
            with st.spinner("🤖 Đang xử lý..."):
                # Phân loại yêu cầu: vẽ biểu đồ hoặc trả lời câu hỏi
                is_visualization_request = any(keyword in prompt.lower() for keyword in 
                                             ["vẽ", "biểu đồ", "chart", "plot", "visualize", "graph", "trực quan"])
                
                if is_visualization_request:
                    if st.session_state.visualization_tool == "Power BI":
                        # Xử lý yêu cầu Power BI
                        powerbi_instructions = f"""
                        ### Hướng dẫn tạo biểu đồ trong Power BI cho yêu cầu: "{prompt}"
                        
                        1. **Tải dữ liệu**: Nhấn nút "📥 Xuất dữ liệu cho Power BI (CSV)" ở sidebar
                        2. **Mở Power BI Desktop** và nhập dữ liệu từ file CSV vừa tải xuống
                        3. **Tạo biểu đồ**:
                        """
                        
                        # Phân tích yêu cầu để đưa ra gợi ý cụ thể cho Power BI
                        try:
                            analysis_prompt = f"""
                            Bạn là chuyên gia về Power BI. Hãy đưa ra hướng dẫn cụ thể để tạo biểu đồ trong Power BI dựa trên yêu cầu sau:
                            
                            Yêu cầu: {prompt}
                            
                            Dữ liệu có các cột: {', '.join(st.session_state.df.columns.tolist())}
                            
                            Hãy đưa ra hướng dẫn từng bước để tạo biểu đồ trong Power BI, bao gồm:
                            1. Loại biểu đồ nên sử dụng
                            2. Các trường dữ liệu nên đặt vào trục nào
                            3. Các bộ lọc nên áp dụng (nếu có)
                            4. Các tính toán DAX cần thiết (nếu có)
                            
                            Chỉ đưa ra hướng dẫn cụ thể, không giải thích lý do.
                            """
                            
                            response = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": "Bạn là chuyên gia về Power BI."},
                                    {"role": "user", "content": analysis_prompt},
                                ],
                                temperature=0.3,
                            )
                            
                            powerbi_steps = response.choices[0].message.content
                            powerbi_instructions += powerbi_steps
                            
                            # Hiển thị hướng dẫn
                            st.markdown(powerbi_instructions)
                            
                            # Tạo nút tải xuống dữ liệu
                            st.download_button(
                                label="📥 Tải dữ liệu cho phân tích này",
                                data=export_for_powerbi(st.session_state.df),
                                file_name="data_for_powerbi_analysis.csv",
                                mime="text/csv"
                            )
                            
                            # Lưu kết quả vào lịch sử
                            result_message = {
                                "role": "assistant", 
                                "content": powerbi_instructions,
                                "chart_type": "powerbi",
                                "chart_data": ""
                            }
                            st.session_state.messages.append(result_message)
                            
                        except Exception as e:
                            error_message = f"❌ Lỗi khi tạo hướng dẫn Power BI: {e}"
                            st.error(error_message)
                            st.session_state.messages.append({"role": "assistant", "content": error_message})
                    
                    else:
                        # Xử lý yêu cầu vẽ biểu đồ bằng Plotly/Matplotlib
                        viz_prompt = build_prompt(prompt, st.session_state.df.head().to_string())
                        
                        try:
                            # Gọi GPT để sinh code
                            response = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": "Bạn là chuyên gia phân tích dữ liệu."},
                                    {"role": "user", "content": viz_prompt},
                                ],
                                temperature=0.3,
                            )
                            
                            # Lấy nội dung trả về
                            raw_code = response.choices[0].message.content
                            
                            # Loại bỏ markdown (```python ... ```) nếu có
                            code_match = re.search(r"```python(.*?)```", raw_code, re.DOTALL)
                            if code_match:
                                code = code_match.group(1).strip()
                            else:
                                # Fallback if no markdown block is found
                                code = raw_code.strip()
                            
                            # Hiển thị code
                            st.code(code, language='python')
                            
                            # Chuẩn bị biến thực thi
                            local_vars = {"df": st.session_state.df, "plt": plt, "px": px}
                            
                            # Thực thi code
                            exec(code, {}, local_vars)
                            
                            # Hiển thị biểu đồ và lưu dữ liệu biểu đồ
                            chart_type = None
                            chart_data = None
                            
                            if "fig" in local_vars:
                                # Xử lý biểu đồ plotly
                                fig = local_vars["fig"]
                                st.plotly_chart(fig, use_container_width=True)
                                chart_type = "plotly"
                                chart_data = plotly_fig_to_json(fig)
                            elif plt.get_fignums():
                                # Xử lý biểu đồ matplotlib
                                fig = plt.gcf()
                                st.pyplot(fig)
                                chart_type = "matplotlib"
                                chart_data = fig_to_base64(fig)
                                plt.clf()
                            else:
                                st.warning("⚠️ Không có biểu đồ nào được tạo.")
                            
                            # Lưu kết quả vào lịch sử
                            result_message = {
                                "role": "assistant", 
                                "content": f"Đây là biểu đồ theo yêu cầu của bạn:\n\n```python\n{code}\n```",
                                "chart_type": chart_type,
                                "chart_data": chart_data
                            }
                            st.session_state.messages.append(result_message)
                            
                        except Exception as e:
                            error_message = f"❌ Lỗi khi thực thi code: {e}"
                            st.error(error_message)
                            st.session_state.messages.append({"role": "assistant", "content": error_message})
                
                else:
                    # Xử lý câu hỏi thông thường
                    try:
                        # Chuẩn bị prompt với thông tin về dữ liệu
                        df_info = f"""
                        Thông tin về dữ liệu:
                        - Số dòng: {st.session_state.df.shape[0]}
                        - Số cột: {st.session_state.df.shape[1]}
                        - Các cột: {', '.join(st.session_state.df.columns.tolist())}
                        - Dữ liệu mẫu (5 dòng đầu):
                        {st.session_state.df.head().to_string()}
                        """
                        
                        qa_prompt = f"""
                        Bạn là trợ lý AI phân tích dữ liệu. Hãy trả lời câu hỏi sau dựa trên dữ liệu được cung cấp:
                        
                        Câu hỏi: {prompt}
                        
                        {df_info}
                        
                        Hãy trả lời ngắn gọn, rõ ràng và cung cấp thông tin hữu ích từ dữ liệu.
                        """
                        
                        # Gọi API để trả lời câu hỏi
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "Bạn là chuyên gia phân tích dữ liệu."},
                                {"role": "user", "content": qa_prompt},
                            ],
                            temperature=0.3,
                        )
                        
                        # Hiển thị câu trả lời
                        answer = response.choices[0].message.content
                        st.markdown(answer)
                        
                        # Lưu câu trả lời vào lịch sử
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                    except Exception as e:
                        error_message = f"❌ Lỗi khi xử lý câu hỏi: {e}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
import sys
import os
from streamlit.web import cli as stcli

if __name__ == "__main__":
    # 현재 실행 파일이 위치한 디렉토리
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # dashboard.py 경로
    dashboard_path = os.path.join(base_path, "esg_src", "dashboard.py")

    # Streamlit 실행
    sys.argv = [
        "streamlit", "run", dashboard_path,
        "--server.headless", "true",
    ]
    sys.exit(stcli.main())



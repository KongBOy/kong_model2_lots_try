print(f"doing {__file__}")
import ctypes, sys
from win32com.shell.shell import ShellExecuteEx
from win32com.shell import shellcon

import win32con, win32event, win32process
import time

if(__name__ == "__main__"):
    if ctypes.windll.shell32.IsUserAnAdmin():  ### 如果是 Administrator 才可做以下的事情
        # 將要執行的程式碼加到這裡
        print('I am elevating to admin privilege.')
        time.sleep(3.5)
        sys.exit()  ### 做完  Administrator 的事情 就可以把這個 terminal 關掉囉！
    else:
        ### 用 Administrator身分 另開一個terminal 執行本程式
        procInfo = ShellExecuteEx(nShow=win32con.SW_SHOWNORMAL,  ### 1
                              fMask=shellcon.SEE_MASK_NOCLOSEPROCESS,  ### 64
                              lpVerb='runas',  ### 'runas'
                              lpFile="%s" % sys.executable,   ### '"C:\\Users\\TKU\\anaconda3\\python.exe"'
                              lpParameters="%s" % __file__  ### '"c:\\Users\\TKU\\Desktop\\try\\trt4.py"'
                              )
        print(procInfo['hProcess'])
        procHandle = procInfo['hProcess']
        obj = win32event.WaitForSingleObject(procHandle, win32event.INFINITE)
        rc = win32process.GetExitCodeProcess(procHandle)
    print("continue")

### 參考網頁：https://newbedev.com/python-python-run-windows-command-as-administrator-code-example
### 參考 try_admin2 才成功
### ctypes.windll.shell32.ShellExecuteW 這個無法等 subprocess做完才繼續做，會直接平行處理
import ctypes, sys
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if is_admin():
    # Code of your program here
    print('I am elevating to admin privilege.')
    input('Press ENTER to exit.')

else:
    # Re-run the program with admin rights
    ctypes.windll.shell32.ShellExecuteW(None, "runas", "%s" % sys.executable, "%s" % __file__, None, 1)

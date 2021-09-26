if(__name__ == "__main__"):
    # print(f"doing {__file__}")
    import ctypes, sys
    from win32com.shell.shell import ShellExecuteEx
    from win32com.shell import shellcon

    import win32con, win32event, win32process

    if ctypes.windll.shell32.IsUserAnAdmin():  ### 如果是 Administrator 才可做以下的事情
        ''' 
        管理員身分執行的程式碼加到這裡
        '''
        print('I am elevating to admin privilege.')
        import os
        git_status = os.system("git clone https://github.com/KongBOy/kong_model2.git")
        if(git_status != 0 ): print("kong_model2 已存在,")
        os.chdir(f"{os.getcwd()}/kong_model2")
        os.system("git submodule init")
        os.system("git submodule update")
        sys.exit()  ### 做完  Administrator 的事情 就可以把這個 terminal 關掉囉！

    else:
        ### 用 Administrator身分 另開一個terminal 執行本程式
        procInfo = ShellExecuteEx(nShow=win32con.SW_SHOWNORMAL,  ### 1
                              fMask=shellcon.SEE_MASK_NOCLOSEPROCESS,  ### 64
                              lpVerb='runas',  ### 'runas'
                              lpFile="%s" % sys.executable,   ### '"C:\\Users\\TKU\\anaconda3\\python.exe"'
                              lpParameters="%s" % __file__  ### '"c:\\Users\\TKU\\Desktop\\try\\trt4.py"'
                              )
        procHandle = procInfo['hProcess']  ### ### <PyHANDLE:1668> 之類的東西
        win32event.WaitForSingleObject(procHandle, win32event.INFINITE)
        win32process.GetExitCodeProcess(procHandle)
    print("continue")

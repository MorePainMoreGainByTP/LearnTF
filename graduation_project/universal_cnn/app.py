import os
import web
from prediction import main
from datetime import datetime
import threading

urls = (
    '/','UploadFile',
)

app = web.application(urls,globals())
web.config.debug = True

render = web.template.render("templates/",cache=False)

class UploadFile:
    """文件上传处理"""
    def handle_pre(self,filedir,filename,index):
        result = main(filedir+"/",filename, index)
        value = result.get(index)
        return value


    def GET(self):
        return render.upload_file()
        # return render.file_upload()

    def POST(self):
        x = web.input(myfile={},operation=[])
        message = ""
        if 'operation' in x:
            print("operation:",x.operation)

        if 'myfile' in x:
            print("myfile:", x.myfile)
            filepath = x.myfile.filename.replace('//', '/')  # 客户端为windows时注意
            filename = filepath.split('/')[-1]  # 获取文件名
            ext = filename.split('.', 1)[1]  # 获取后缀
            if ext == 'jpg':  # 判断文件后缀名
                homedir = os.getcwd()
                filedir = '%s/static/uploads' % homedir  # 要上传的路径
                now = datetime.now()
                t = "%d%d%d%d%d%d" % (now.year, now.month, now.day, now.hour, now.minute, now.second)  # 以时间作为文件名
                filename = t + '.' + ext
                fout = open(filedir + '/' + filename, 'wb')
                fout.write(x.myfile.file.read())
                fout.close()
                res = {}
                mes = ""
                for index in x.operation:
                    res[index] = self.handle_pre(filedir,filename,index)
                for key,value in res.items():
                    if key == "1":
                        if value == 1:
                            mes += " the picture through Gaussion Blur Operation\n"
                        else:
                            mes += " the picture not through Gaussion Blur Operation\n"
                    elif key == "2":
                        if value == 1:
                            mes += " the picture through Surface Blur Operation\n"
                        else:
                            mes += " the picture not through Surface Blur Operation\n"
                    elif key == "3":
                        if value == 1:
                            mes += " the picture through AWGN(Additive White Gaussian Noise) Operation\n"
                        else:
                            mes += " the picture not through AWGN(Additive White Gaussian Noise) Operation\n"
                    elif key == "4":
                        if value == 1:
                            mes += " the picture through Sharpen Operation\n"
                        else:
                            mes += " the picture not through Sharpen Operation\n"
                message = u'' + "\nthe prediction result:" + mes
                error = False
            else:
                message = u'请上传jpg格式的文件!'
                error = True
        return render.result(message)


if __name__ == "__main__":
    app.run()
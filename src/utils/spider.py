# coding=utf-8

import cookielib
import urllib

url = "http://www.baidu.com"
response1 = urllib.urlopen(url)
print("第一种方法")
# 获取状态码，200表示成功
print(response1.getcode())
# 获取网页内容的长度
print(len(response1.read()))

print("第二种方法")
request = urllib.Request(url)
# 模拟Mozilla浏览器进行爬虫
request.add_header("user-agent", "Mozilla/5.0")
response2 = urllib.urlopen(request)
print(response2.getcode())
print(len(response2.read()))

print("第三种方法")
cookie = cookielib.CookieJar()
# 加入urllib2处理cookie的能力
opener = urllib.build_opener(urllib.HTTPCookieProcessor(cookie))
urllib.install_opener(opener)
response3 = urllib.urlopen(url)
print(response3.getcode())
print(len(response3.read()))
print(cookie)

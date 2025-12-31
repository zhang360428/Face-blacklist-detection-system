# 人脸识别项目

该项目已经部署为FastAPI的形式，地址在http://127.0.0.1:9876
需要新建黑人脸库或者是需要向其中新增图片请参考build_new_face_library.py代码
该项目依赖于milvus数据库，如果docker容器宕机请在/mnt/data4/dcr/大模型网关/人脸识别/src/milvus-docker目录下输入docker-compose up -d即可重新开启容器
该项目对应的conda环境为"FaceSafety"
需要启动黑人脸库服务请运行face_detection.py或者是使用./ start_service.py来启动服务

对外提供的接口可参考test_service.py中的接口，也可以直接使用curl命令来访问我们的黑人脸检测服务，示例如下：
curl -X POST "http://127.0.0.1:9876/detect" -H "Content-Type: application/json" -d '{"image_path": "/mnt/data4/dcr/大模型网关/人脸识别/src/FaceSafety/Input/20251230_test01/0001.jpeg"}'

注意只需要提供图片的绝对地址即可，jpg,jpeg.png,bmp等格式均支持

构建黑人脸库所用的目录在/mnt/data4/dcr/大模型网关/人脸识别/src/FaceSafety/Input/20251230_01
检测可以用数据增强后或者是达赖喇嘛的其他图片来测试，地址在/mnt/data4/dcr/大模型网关/人脸识别/src/FaceSafety/Input/20251230_test01


文件检测三种方式：

## 通过图片路径检测 (/detect)
直接传入图片的绝对路径进行检测
curl -X POST "http://127.0.0.1:9876/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/mnt/data4/dcr/测试图片/嫌疑人张三_001.jpg"
  }'

## 通过文件上传检测 (/upload)
使用 multipart/form-data 格式上传图片文件
curl -X POST "http://127.0.0.1:9876/upload" \
  -F "file=@/mnt/data4/dcr/测试图片/嫌疑人李四_002.jpg"

## 通过Base64编码检测 (/detect_base64)
传入Base64编码的图片数据，支持纯Base64和Data URI格式
使用纯Base64（需要先编码）
base64_image=$(base64 -w 0 /mnt/data4/dcr/测试图片/未知人员_003.jpg)

curl -X POST "http://127.0.0.1:9876/detect_base64" \
  -H "Content-Type: application/json" \
  -d "{
    \"image_base64\": \"$base64_image\",
    \"filename\": \"unknown_person.jpg\"
  }"

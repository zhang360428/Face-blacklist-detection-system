# test_service.py
import requests
import sys
import json

def test_service():
    """测试人脸检测服务"""
    
    # 测试用的图片路径
    test_image_path = "path/to/FaceSafety/Input/20251230_test01/0001.jpeg"
    
    print("="*60)
    print("测试人脸检测服务")
    print("="*60)
    
    # 测试1: 健康检查
    print("\n1. 健康检查...")
    try:
        response = requests.get("http://127.0.0.1:9876/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
    except Exception as e:
        print(f"健康检查失败: {str(e)}")
        return False
    
    # 测试2: 统计信息
    print("\n2. 获取统计信息...")
    try:
        response = requests.get("http://127.0.0.1:9876/stats")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
    except Exception as e:
        print(f"获取统计信息失败: {str(e)}")
        return False
    
    # 测试3: 人脸检测
    print(f"\n3. 检测图片: {test_image_path}")
    try:
        response = requests.post(
            "http://127.0.0.1:9876/detect",
            json={"image_path": test_image_path}
        )
        
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200:
            if result["detected"]:
                if result["predicted_in_blacklist"]:
                    print("\n✅ 结果: 人脸在黑名单中")
                    print(f"匹配人名: {result['matched_person']}")
                    print(f"相似度: {result['similarity']:.4f}")
                else:
                    print("\n✅ 结果: 人脸不在黑名单中")
                    print(f"最高相似度: {result['similarity']:.4f} (低于阈值 {result['threshold']})")
            else:
                print("\n⚠️ 未检测到人脸")
        else:
            print(f"\n❌ 请求失败: {result}")
            return False
            
    except Exception as e:
        print(f"检测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_service()
    sys.exit(0 if success else 1)

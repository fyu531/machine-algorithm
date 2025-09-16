import http.server
import socketserver
import json
import os
from urllib.parse import urlparse, parse_qs
import importlib
import random

# 定义服务器端口
PORT = 8000

# 数据集和算法映射
DATASETS = {
    'iris': 'datasets.iris',
    'mnist': 'datasets.mnist_sample',
    'regression': 'datasets.regression_sample'
}

ALGORITHMS = {
    'decision_tree': 'algorithms.decision_tree',
    'naive_bayes': 'algorithms.naive_bayes',
    'knn': 'algorithms.knn',
    'svm': 'algorithms.svm',
    'random_forest': 'algorithms.random_forest',
    'linear_regression': 'algorithms.linear_regression',
    'logistic_regression': 'algorithms.logistic_regression',
    'adaboost': 'algorithms.adaboost',
    'kmeans': 'algorithms.kmeans',
    'em': 'algorithms.em'
}

class MLRequestHandler(http.server.SimpleHTTPRequestHandler):
    """处理机器学习相关的HTTP请求"""
    
    def _set_headers(self, status_code=200, content_type='application/json'):
        """设置HTTP响应头"""
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')  # 允许跨域请求
        self.end_headers()
    
    def do_GET(self):
        """处理GET请求"""
        parsed_url = urlparse(self.path)
        
        # 提供前端静态文件
        if parsed_url.path.startswith('/frontend/'):
            # 调整路径以指向实际的前端文件目录
            self.path = parsed_url.path.replace('/frontend/', '../frontend/')
            return super().do_GET()
        
        # 获取数据集
        elif parsed_url.path.startswith('/dataset/'):
            dataset_name = parsed_url.path.split('/')[2]
            self.handle_get_dataset(dataset_name)
        
        # 获取算法信息
        elif parsed_url.path.startswith('/algorithm/'):
            algorithm_name = parsed_url.path.split('/')[2]
            self.handle_get_algorithm(algorithm_name)
        
        # 未知路径
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'Not found'}).encode())
    
    def do_POST(self):
        """处理POST请求"""
        parsed_url = urlparse(self.path)
        
        # 运行算法
        if parsed_url.path == '/run-algorithm':
            self.handle_run_algorithm()
        
        # 未知路径
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'Not found'}).encode())
    
    def handle_get_dataset(self, dataset_name):
        """处理获取数据集的请求"""
        if dataset_name not in DATASETS:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'Dataset not found'}).encode())
            return
        
        try:
            # 动态导入数据集模块
            dataset_module = importlib.import_module(DATASETS[dataset_name])
            data = dataset_module.load_data()
            
            self._set_headers(200)
            self.wfile.write(json.dumps({
                'name': dataset_name,
                'description': dataset_module.DESCRIPTION,
                'data': data
            }).encode())
        
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({'error': str(e)}).encode())
    
    def handle_get_algorithm(self, algorithm_name):
        """处理获取算法信息的请求"""
        if algorithm_name not in ALGORITHMS:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'Algorithm not found'}).encode())
            return
        
        try:
            # 动态导入算法模块
            algorithm_module = importlib.import_module(ALGORITHMS[algorithm_name])
            
            self._set_headers(200)
            self.wfile.write(json.dumps({
                'name': algorithm_name,
                'description': algorithm_module.DESCRIPTION,
                'parameters': algorithm_module.PARAMETERS
            }).encode())
        
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({'error': str(e)}).encode())
    
    def handle_run_algorithm(self):
        """处理运行算法的请求"""
        # 读取请求体
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode())
        
        # 验证请求数据
        if 'algorithm' not in data or 'dataset' not in data:
            self._set_headers(400)
            self.wfile.write(json.dumps({'error': 'Missing algorithm or dataset'}).encode())
            return
        
        algorithm_name = data['algorithm']
        dataset_name = data['dataset']
        parameters = data.get('parameters', {})
        
        if algorithm_name not in ALGORITHMS or dataset_name not in DATASETS:
            self._set_headers(400)
            self.wfile.write(json.dumps({'error': 'Invalid algorithm or dataset'}).encode())
            return
        
        try:
            # 加载数据集
            dataset_module = importlib.import_module(DATASETS[dataset_name])
            dataset = dataset_module.load_data()
            
            # 导入并运行算法
            algorithm_module = importlib.import_module(ALGORITHMS[algorithm_name])
            result = algorithm_module.run(dataset, parameters)
            
            self._set_headers(200)
            self.wfile.write(json.dumps(result).encode())
        
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({'error': str(e)}).encode())

def run_server():
    """启动服务器"""
    handler = MLRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"服务器启动在端口 {PORT}")
        print(f"访问 http://localhost:{PORT}/frontend/index.html 查看界面")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器正在关闭...")
            httpd.shutdown()

if __name__ == "__main__":
    run_server()


from flask import Flask, request, jsonify
from model.inference import Inference

app = Flask(__name__)

# Initialize the Inference class
inference_instance = Inference()

class API:
    def __init__(self):
        self.data = request.json
    @app.route('/api/db', methods=['POST'])
    def create_vector_db(self):
        symbol = self.data.get('symbol')

        if not symbol:
            return jsonify({"error": "Symbol is required."}), 400
        embeddings = inference_instance.load_embeddings()
        self.retreiver = inference_instance.get_vectorspace(symbol=symbol,embeddings=embeddings)
        del embeddings
        return jsonify({"message": f"Vector database created for symbol: {symbol}"}), 201

    @app.route('/api/resp', methods=['POST'])
    def inference(self):
        question = self.data.get('question')

        if not question:
            return jsonify({"error": "question are required."}), 400

        # Get the response from the Inference class
        context = inference_instance.get_context(question=question)
        resp = inference_instance.inference(context=context,question=question)
        response = inference_instance.format_output(inference_instance.answer_extract(resp))
        return jsonify({"response": response})

    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "running"}), 200

if __name__ == '__main__':
    app.run(debug=True)

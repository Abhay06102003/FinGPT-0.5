from flask import Flask, request, jsonify
from model.inference import Inference
import asyncio

app = Flask(__name__)

# Initialize the Inference class
inference_instance = Inference()

class API:
    def __init__(self):
        self.data = request.json

    @app.route('/api/db', methods=['POST'])
    def create_vector_db():
        data = request.json  # Move request data inside the method
        symbol = data.get('symbol')

        if not symbol:
            return jsonify({"error": "Symbol is required."}), 400
        embeddings = inference_instance.load_embeddings()
        inference_instance.retriever = inference_instance.get_vectorspace(symbol=symbol, embeddings=embeddings)
        del embeddings
        return jsonify({"message": f"Vector database created for symbol: {symbol}"}), 201

    @app.route('/api/resp', methods=['POST'])
    async def inference():
        data = request.json  # Move request data inside the method
        question = data.get('question')

        if not question:
            return jsonify({"error": "Question is required."}), 400

        # Get the context asynchronously
        context = await asyncio.get_event_loop().run_in_executor(None, inference_instance.get_context, question)
        # Get the response from the Inference class
        resp = await asyncio.get_event_loop().run_in_executor(None, inference_instance.inference, context, question)
        response = inference_instance.format_output(inference_instance.answer_extract(resp))
        return jsonify({"response": response})

    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "running"}), 200

if __name__ == '__main__':
    app.run(debug=True)

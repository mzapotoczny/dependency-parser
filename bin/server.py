import json
import logging
import jsonschema
import tornado.ioloop
import tornado.web
from datetime import datetime, timedelta
from tornado.queues import Queue 
from tornado import gen
from tornado.gen import TimeoutError
from tornado.concurrent import Future
from lvsr.parser import get_parser

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger("Server")
logger.setLevel(logging.INFO)

taskQueue = Queue()
timeoutSeconds = 0.5
queueSize = 16

class DependencyHandler(tornado.web.RequestHandler):
    SUPPORTED_METHODS = ("POST")
    
    schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "decoder": {"type": "string", "enum" : ["greedy", "nonproj"]},
        },
        "required": ["text"],
        "additionalProperties": False
    }
    
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, OPTIONS')

    def post(self):
        error = None
        try:
            json_data = json.loads(self.request.body)
            jsonschema.validate(json_data, self.schema)
        except ValueError as parseError:
            error = "Parse error: {}".format(parseError.message)
        except jsonschema.exceptions.ValidationError as validationError:
            error = "Validation error: {}".format(validationError.message)
            
        if error is None:
            resp = Future()
            taskQueue.put((resp, json_data))
            resp.add_done_callback(lambda x: self.write(x.result()))
            logger.info(u"Got sentence: {}".format(json_data['text']))
            return resp 
        else:
            logger.info(u"Wrong request, reason: {}".format(error))
            self.write({"error": error})

@gen.coroutine 
def parseWorker():
    timeout = timedelta(seconds=timeoutSeconds)
    currentTimeout = timeout
    tasks = []
    while True:
        task = yield taskQueue.get()
        tasks.append(task)
        
        while currentTimeout.total_seconds() > 0 and len(tasks) < queueSize:
            start = datetime.now()
            try:
                task = yield taskQueue.get(timeout=currentTimeout)
                tasks.append(task)
                currentTimeout -= datetime.now() - start
            except TimeoutError:
                break
            
        parser_input = [task[1]['text'] for task in tasks]
        parser_decoders = [task[1]['decoder'] if 'decoder' in task[1] else None for task in tasks]

        error = False
        try:
            logger.info(u"Parsing {} sentences. Max length {}"
                    .format(len(parser_input), max([len(x) for x in parser_input])))
            outputs = dependencyParser(parser_input, parser_decoders)
        except:
            error = True
        for i, task in enumerate(tasks):
            if error:
                result = {'error': 'An error occured during parsing'}
            else:
                result = {'conll': outputs[i]}
            task[0].set_result(result)
        
        currentTimeout = timeout
        tasks = []

application = tornado.web.Application([
    (r"/", DependencyHandler)
])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    # TODO: lang can be dynamic
    parser.add_argument('--lang', default=None)
    parser.add_argument('--logfile', default=None)
    parser.add_argument('--port', type=int, default=8888)
    parser.add_argument('--queuesize', type=int, default=16)
    parser.add_argument('--timeout', type=float, default=0.5)

    args = parser.parse_args()
    model = args.model
    lang = args.lang
    port = args.port

    if args.logfile:
        fileHandler = logging.FileHandler(args.logfile)
        fileHandler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(fileHandler)

    queueSize = args.queuesize
    timeoutSeconds = args.timeout

    logger.info("Initialization started")
    dependencyParser = get_parser(model, None, lang)
    logger.info("Parser initialized. Ready")

    application.listen(port)
    io_loop = tornado.ioloop.IOLoop.current()
    io_loop.run_sync(parseWorker)

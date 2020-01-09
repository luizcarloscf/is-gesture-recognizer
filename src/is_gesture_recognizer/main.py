import re
import time
import dateutil.parser as dp

from prometheus_client import start_http_server, Gauge
from is_wire.core import Subscription, Message, Channel, Logger, Tracer, AsyncTransport
from opencensus.ext.zipkin.trace_exporter import ZipkinExporter
from is_msgs.image_pb2 import ObjectAnnotations

from utils import load_options
from gesture import GestureRecognizer
from skeleton import Skeleton


def span_duration_ms(span):
    """ Funtion to measure the time of a span in the Zipkin instrumentation
    """
    dt = dp.parse(span.end_time) - dp.parse(span.start_time)
    return dt.total_seconds() * 1000.0


def create_exporter(service_name, uri):
    """ Funtion to create the exporter in the Zipkin instrumentation
    """
    log = Logger(name="CreateExporter")
    zipkin_ok = re.match("http:\\/\\/([a-zA-Z0-9\\.]+)(:(\\d+))?", uri)
    if not zipkin_ok:
        log.critical("Invalid zipkin uri \"{}\", expected http://<hostname>:<port>", uri)
    exporter = ZipkinExporter(service_name=service_name,
                              host_name=zipkin_ok.group(1),
                              port=zipkin_ok.group(3),
                              transport=AsyncTransport)
    return exporter


def main():

    # Defining our service
    service_name = 'GestureRecognizier.Recognition'
    log = Logger(name=service_name)

    # Loading options
    op = load_options()

    # Conecting to the broker
    channel = Channel(op.broker_uri)
    log.info('Connected to broker {}', op.broker_uri)

    # creating the Zipking exporter
    exporter = create_exporter(service_name=service_name, uri=op.zipkin_uri)

    # Subcripting on desired topics
    subscription = Subscription(channel=channel, name=service_name)
    for group_id in list(op.group_ids):
        subscription.subscribe('SkeletonsGrouper.{}.Localization'.format(group_id))

    # initialize the Model
    model = GestureRecognizer()
    log.info('Initialize the model')

    # load the model
    model.load("model_gesture3_0_91.16.pth")

    # metrics for monitoring the system
    unc = Gauge('uncertainty', "Uncertainty about the predict")

    # default values of the metrics

    unc.set(0)

    # starting the server
    start_http_server(8000)

    # list and time to take the median
    buffer = list()
    initial_time = time.time()

    # begining the service
    while True:

        # waiting for receive a message
        msg = channel.consume()

        # initialize the Tracer
        tracer = Tracer(exporter, span_context=msg.extract_tracing())
        span = tracer.start_span(name='detection_and_info')
        detection_span = None

        # unpack the message
        with tracer.span(name='unpack'):
            annotations = msg.unpack(ObjectAnnotations)
            skeletons = [Skeleton(obj) for obj in annotations.objects]
            if len(skeletons) > 0:
                skl = skeletons[0]
                skl_normalized = skl.normalize()
                skl_vector = skl_normalized.vectorized()

        # preditic
        with tracer.span(name='detection') as _span:
            if len(skeletons) > 0:
                pred, prob, uncertainty = model.predict(skl_vector)
                detection_span = _span

        # finish the tracer
        tracer.end_span()

        # update metrics values
        if len(skeletons) > 0:
            if (time.time() - initial_time) > op.period and len(buffer) > 0:
                unc.set(sum(buffer) / len(buffer))
                buffer = []
                initial_time = time.time()

            else:
                buffer.append(uncertainty)

            # logging usefull informations
            info = {
                'prediction': pred,
                'probability': prob,
                'uncertainty': uncertainty,
                'took_ms': {
                    'detection': round(span_duration_ms(detection_span), 2),
                    'service': round(span_duration_ms(span), 2)
                }
            }
            log.info('{}', str(info).replace("'", '"'))


if __name__ == "__main__":
    main()
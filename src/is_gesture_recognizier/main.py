import re
import dateutil.parser as dp

from is_wire.core import Subscription, Message, Channel, Logger, Tracer, AsyncTransport
from opencensus.ext.zipkin.trace_exporter import ZipkinExporter
from is_msgs.image_pb2 import ObjectAnnotations

from utils import load_options
from model import Model


def span_duration_ms(span):
    dt = dp.parse(span.end_time) - dp.parse(span.start_time)
    return dt.total_seconds() * 1000.0


def create_exporter(service_name, uri):

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

    #name of the service
    service_name = 'GestureRecognizier.Recognition'

    #logging info
    log = Logger(name=service_name)

    #loading options
    op = load_options()

    #regex object to match the
    re_topic = re.compile(r'SkeletonsGrouper.(\w+).Localization')
    channel = Channel(op.broker_uri)
    log.info('Connected to broker {}', op.broker_uri)

    exporter = create_exporter(service_name=service_name, uri=op.zipkin_uri)

    subscription = Subscription(channel=channel, name=service_name)

    for group_id in list(op.group_ids):
        subscription.subscribe('SkeletonsGrouper.{}.Localization'.format(group_id))

    while True:

        msg = channel.consume()

        tracer = Tracer(exporter, span_context=msg.extract_tracing())
        span = tracer.start_span(name='detection_and_info')

        with tracer.span(name='unpack'):
            annotations = msg.unpack(ObjectAnnotations)
            skeletons = [obj for obj in annotations.objects]

        tracer.end_span()

        info = {
            'detections': len(skeletons),
        }

        log.info('{}', str(info).replace("'", '"'))


if __name__ == "__main__":
    main()
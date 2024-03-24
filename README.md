# Network Intrusion Detection System

A Network Intrusion Detection System (NIDS) implemented using Streamlit and TensorFlow/Keras. This system predicts whether a network connection is normal or abnormal based on various features of the connection.

## Features

The following features are used for prediction:

- `protocol_type`: The protocol type of the connection (e.g., TCP, UDP, ICMP).
- `flag`: The status of the connection (e.g., FIN, SYN, RST).
- `service`: The network service on the destination (e.g., http, ftp, telnet).
- `src_bytes`: The number of bytes sent by the source.
- `dst_bytes`: The number of bytes sent to the destination.
- `count`: The number of connections to the same destination host.
- `same_srv_rate`: The percentage of connections to the same service.
- `diff_srv_rate`: The percentage of connections to different services.
- `dst_host_srv_count`: The number of connections to the same destination host and service.
- `dst_host_same_srv_rate`: The percentage of connections to the same service on the same destination host.
- `dst_host_same_src_port_rate`: The percentage of connections from the same source port to the same destination host.



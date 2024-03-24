import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from 

# Load data
@st.cache_data
def load_data():
    train = pd.read_csv('Train_data.csv')
    test = pd.read_csv('Test_data.csv')
    return train, test

train, test = load_data()

# Function to preprocess data
def le(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            df[col+'_num'] = label_encoder.fit_transform(df[col])
    return df

# Apply label encoding to both train and test data
train = le(train)
test = le(test)

# Streamlit UI
def main():
    nav = st.sidebar.radio("Navigation",["Readme","Prediction"])
    if nav== "Readme":
        st.write("Features:")
        features = {
                'protocol_type': 'The protocol type of the connection (e.g., TCP, UDP, ICMP). Different protocols may be associated with different types of attacks.',
                'flag': 'The status of the connection (e.g., FIN, SYN, RST). Unusual flag combinations may indicate suspicious activity like port scanning.',
                'service': 'The network service on the destination (e.g., http, ftp, telnet). Certain services may be targeted more frequently by attackers.',
                'src_bytes': 'The number of bytes sent by the source.',
                'dst_bytes': 'The number of bytes sent to the destination.',
                'count': 'The number of connections to the same destination host.',
                'same_srv_rate': 'The percentage of connections to the same service. Unusually high or low rates may indicate scanning or targeted attacks.',
                'diff_srv_rate': 'The percentage of connections to different services.',
                'dst_host_srv_count': 'The number of connections to the same destination host and service.',
                'dst_host_same_srv_rate': 'The percentage of connections to the same service on the same destination host.',
                'dst_host_same_src_port_rate': 'The percentage of connections from the same source port to the same destination host.'
        }
        for feature, description in features.items():
            st.write(f"- {feature}: {description}")

        history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    if nav == "Prediction":
        st.title('Network Intrusion Detection System')  
        st.write("## Input Values for Features")

        protocol_type = st.selectbox('protocol_type', train['protocol_type'].unique())
        duration = st.number_input('duration')
        flag = st.selectbox('flag', train['flag'].unique())
        service = st.selectbox('service', train['service'].unique())
        src_bytes = st.number_input('src_bytes')
        dst_bytes = st.number_input('dst_bytes')
        dst_host_srv_count = st.number_input('dst_host_srv_count')
        dst_host_count = st.number_input('dst_host_count')
        dst_host_same_src_port_rate = st.number_input('dst_host_same_src_port_rate')
        srv_serror_rate = st.number_input('srv_serror_rate')

        if st.button('Process Data'):
            # Create a dictionary with selected values
            input_data = {
                'src_bytes': [src_bytes],
                'dst_bytes': [dst_bytes],
                'duration': [duration],
                'dst_host_srv_count': [dst_host_srv_count],
                'dst_host_count': [dst_host_count],  # Ensure 'dst_host_count' is included here
                'service': [service], 
                'flag': [flag],
                'protocol_type': [protocol_type],
                'srv_serror_rate': [srv_serror_rate],
                'dst_host_same_src_port_rate': [dst_host_same_src_port_rate]
            }

            # Convert dictionary to DataFrame
            input_df = pd.DataFrame(input_data)
            input_df = le(input_df)
            prediction = model.predict(input_df[['src_bytes', 'dst_bytes', 
                                                'duration', 'dst_host_srv_count', 'dst_host_count', 'service_num', 'flag_num', 
                                                'srv_serror_rate', 'protocol_type_num', 'dst_host_same_src_port_rate']])
            # Display prediction
            # y_pred = (prediction > 0.5).astype(int)
            # st.write("## Prediction")
            # st.write(y_pred)
            if prediction > 0.5:
                st.write('The system predicts: Normal')
            else:
                st.write('The system predicts: Abnormal')


if  __name__== "__main__":
    main()
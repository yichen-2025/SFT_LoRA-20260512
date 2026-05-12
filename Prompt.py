
def feature_to_natural_language(row):
    """
    将数值特征转换为自然语言描述
    params:
        row: 包含数值特征的行数据
    return:
        自然语言描述
    """
    descriptions="你是一个网络安全专家，你需要判断以下网络流量是否为恶意流量。网络流量的特征如下："

    # Average Packet Size
    average_packet_size=row['Average Packet Size']
    descriptions+=f"平均包大小为{average_packet_size}字节。"

    # Packet Length Mean
    packet_length_mean=row['Packet Length Mean']
    descriptions+=f"包长度均值为{packet_length_mean}字节。"

    # Packet Length Std
    packet_length_std=row['Packet Length Std']
    descriptions+=f"包长度标准差为{packet_length_std}字节。"

    # Avg Fwd Segment Size
    avg_fwd_segment_size=row['Avg Fwd Segment Size']
    descriptions+=f"平均前向段大小为{avg_fwd_segment_size}字节。"

    # Avg Bwd Segment Size
    avg_bwd_segment_size=row['Avg Bwd Segment Size']
    descriptions+=f"平均后向段大小为{avg_bwd_segment_size}字节。"

    # Bwd Packet Length Mean
    bwd_packet_length_mean=row['Bwd Packet Length Mean']
    descriptions+=f"后向包长度均值为{bwd_packet_length_mean}字节。"

    # Bwd Packet Length Std
    bwd_packet_length_std=row['Bwd Packet Length Std']
    descriptions+=f"后向包长度标准差为{bwd_packet_length_std}字节。"

    # Bwd ratio
    total_fwd_packets=row['Total Fwd Packets']
    total_bwd_packets=row['Total Backward Packets']
    bwd_ratio=total_bwd_packets/(total_fwd_packets+total_bwd_packets)
    descriptions+=f"后向包比例为{bwd_ratio}。"

    # SYN ratio
    total_syn_packets=row['SYN Flag Count']
    syn_ratio=total_syn_packets/(total_fwd_packets+total_bwd_packets)
    descriptions+=f"SYN包比例为{syn_ratio}。"

    # Flow IAT Std
    flow_iat_std=row['Flow IAT Std']
    descriptions+=f"流间隔时间标准差为{flow_iat_std}秒。"

    descriptions+="请根据以上特征判断网络流量是否为恶意流量。只要回答“正常流量”或“恶意流量”即可。"

    return descriptions

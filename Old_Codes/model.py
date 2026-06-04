class v9_rescaling_adaptive_TransformerModel3Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead, num_layers, max_len=150, dropout=0.5):
        super(v9_rescaling_adaptive_TransformerModel3Decoder, self).__init__()
        self.linear_in = nn.Linear(input_dim+1, hidden_dim)
        self.linear_in_1= nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(input_dim)  
        self.layer_norm_final = nn.LayerNorm(output_dim)  
        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, hidden_dim))

        # Transformer Decoder layers
        decoder_layers = nn.TransformerDecoderLayer(d_model=2*hidden_dim, nhead=nhead, dropout=dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.scaling_factors_0th = nn.Parameter(torch.ones(1,max_len))
        self.scaling_factors_1st = nn.Parameter(torch.ones(1,max_len))


        # Output linear layer
        self.linear_out1 = nn.Linear(2hidden_dim, 5output_dim)
        self.linear_out2 = nn.Linear(5output_dim, 2output_dim)
        self.linear_out3 = nn.Linear(2output_dim,1output_dim)
        self.conv1d_volt = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv1d_temp = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

        

        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, initial_state, actions):
        # Repeat initial state to match action sequence length
        original_initial_state = initial_state.clone()
        initial_state[:,1]= initial_state[:,1]/3
        initial_state[:,2]= initial_state[:,2]/30

        actions = (actions)/5
        # print(actions.shape,"#######")
        actions_clone_1= actions.clone()
        actions_clone_2= actions.clone()
        actions_delta_shifted = actions_clone_2
        actions_delta_shifted[:,0:-1,:]-=actions_clone_1[:,1:,:]
        actions_delta_shifted[:,-1,:]=0
        
        #actions_delta_shifted= actions_delta_shifted.to(actions.device)
        # print(actions_delta_shifted[:,0:-1,:].shape,actions_clone_1[:,1:,:].shape,"####")
        # print(actions_delta_shifted)
        # print(jd)

        power_=10*(actions**2)# It is also scaled

        repeated_original_initial_state = original_initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)

        initial_state_repeated = initial_state.unsqueeze(1).repeat(1, actions.size(1), 1)
        #print(initial_state_repeated.device, actions.device,actions_delta_shifted.device,"###")
        transformer_input = torch.cat([initial_state_repeated, actions,actions_delta_shifted], dim=-1)
        transformer_input = self.linear_in(transformer_input)
        transformer_input = self.linear_in_1(transformer_input)
        
        pos_encoding = self.positional_encoding[:transformer_input.size(1), :]
        pos_encoding_expanded = pos_encoding.unsqueeze(0).expand(transformer_input.size(0), -1, -1)

        transformer_input = torch.cat((transformer_input, pos_encoding_expanded), dim=-1)

        tgt_mask = self.generate_square_subsequent_mask(actions.size(1)).to(actions.device)
        transformer_output = self.transformer_decoder(transformer_input, transformer_input, tgt_mask=tgt_mask)

        predicted_states_residual_1 = self.linear_out1(transformer_output)
        predicted_states_residual_1 = self.gelu(predicted_states_residual_1)
        predicted_states_residual_2 = self.linear_out2(predicted_states_residual_1)
        predicted_states_residual_2 = self.gelu(predicted_states_residual_2)
        predicted_states_residual_3 = self.linear_out3(predicted_states_residual_2)


        # spacing_array = torch.linspace(1, 15, 150).to(actions.device).unsqueeze(0)
        # #current_based_spacing_temporal = actions.squeeze(dim=2)*spacing_array
        # current_based_spacing_temporal = actions_delta_shifted.squeeze(dim=2)*spacing_array
        # power_based_spacing_temporal = power_.squeeze(dim=2)*spacing_array
        # #############
        # # v9 had convolution v10 won't
        # current_based_spacing_temporal = current_based_spacing_temporal.unsqueeze(1)  # Add channel dimension
        # current_based_spacing_temporal = self.conv1d_volt(current_based_spacing_temporal)
        # current_based_spacing_temporal = current_based_spacing_temporal.squeeze(1)  # Remove channel dimension
        
        # power_based_spacing_temporal = power_based_spacing_temporal.unsqueeze(1)  # Add channel dimension
        # power_based_spacing_temporal = self.conv1d_temp(power_based_spacing_temporal)
        # power_based_spacing_temporal = power_based_spacing_temporal.squeeze(1)  # Remove channel dimension
        #############

        # print(current_based_spacing_temporal.shape,power_based_spacing_temporal.shape)
        # print(predicted_states_residual_3.shape)
        # print(jd)

        repeated_original_voltage = repeated_original_initial_state[:,:,1]
        repeated_original_temprature = repeated_original_initial_state[:,:,2]

        predicted_states_voltage = repeated_original_voltage + predicted_states_residual_3[:,:,0]
        predicted_states_temperature = repeated_original_temprature+ (predicted_states_residual_3[:,:,1]/10)
        
        
        predicted_states_final = torch.stack((predicted_states_voltage, predicted_states_temperature), dim=2)


        return predicted_states_final

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
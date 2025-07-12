def parse_tecplot_data(file_path):
    print(f'Parsing file: {file_path}...')
    
    headers = []
    data = []
    variables_line = []
    cumulative_time = 0.0

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Step 1: Parse the VARIABLES header
    for i, line in enumerate(lines):
        if "variables" in line.lower() or variables_line:
            variables_line.append(line.strip())
            if not line.strip().endswith("\\"):  # End of VARIABLES section
                full_variables = " ".join(variables_line).replace("\\", "").replace("VARIABLES =", "").strip()
                headers = [col.strip().strip('"') for col in full_variables.split(",")]
                data_start_index = i + 1
                break

    if not headers:
        raise ValueError("VARIABLES section not found in the file. Please check the input file format.")
    
    print(f"Headers: {headers}")
    
    # Step 2: Parse the data
    for line in lines[data_start_index:]:
        if not line.strip():  # Skip empty lines
            continue
        
        try:
            # Handle comma-separated values
            values = [float(val) for val in line.split(",")]
            data.append(values)
        except ValueError as e:
            print(f"Skipping line due to error: {e}")
            continue

    print(f"Parsed {len(data)} rows of data.")

    # Step 3: Add a cumulative time column
    if "Time(sec)" not in headers:
        raise ValueError("The column 'Time(sec)' is not in the headers.")
    
    time_index = headers.index("Time(sec)")
    processed_data = []
    for row in data:
        cumulative_time += row[time_index]
        processed_data.append(row + [cumulative_time])
    
    headers.append("Cumulative_Time")
    print(f"Added cumulative time column. Total time: {cumulative_time:.6f}")

    return variables_line, headers, processed_data


def save_tecplot_data(output_file, variables_line, headers, data):
    print(f"Saving Tecplot-compatible data to: {output_file}...")
    with open(output_file, 'w') as file:
        # Write the VARIABLES line
        file.write("VARIABLES = \\\n")
        for i, header in enumerate(headers):
            file.write(f'"{header}"')
            if i < len(headers) - 1:
                file.write(",\n")
            else:
                file.write("\n")
        
        # Write the data rows
        for row in data:
            file.write(" ".join(f"{value:.6f}" for value in row) + "\n")
    
    print("Data saved successfully in Tecplot format.")


# Main execution
# input_file = "/home/jnm8/Desktop/512/flow_conv_.dat"
# output_file = "/home/jnm8/Desktop/512/flow_conv2_.dat"

input_file = "/home/jnm8/Desktop/logs/cores-test/64/flow_conv_.dat"
output_file = "/home/jnm8/Desktop/logs/cores-test/64/flow_conv_64.dat"

# input_file = "/home/jnm8/Desktop/logs/cores-test/128/flow_conv_.dat"
# output_file = "/home/jnm8/Desktop/logs/cores-test/128/flow_conv_128.dat"

# input_file = "/home/jnm8/Desktop/logs/cores-test/192/flow_conv_.dat"
# output_file = "/home/jnm8/Desktop/logs/cores-test/192/flow_conv_192.dat"

# input_file = "/home/jnm8/Desktop/logs/cores-test/256/flow_conv_.dat"
# output_file = "/home/jnm8/Desktop/logs/cores-test/256/flow_conv_256.dat"

# input_file = "/home/jnm8/Desktop/logs/cores-test/320/flow_conv_.dat"
# output_file = "/home/jnm8/Desktop/logs/cores-test/320/flow_conv_320.dat"

# input_file = "/home/jnm8/Desktop/logs/cores-test/512/250GB_total/flow_conv_.dat"
# output_file = "/home/jnm8/Desktop/logs/cores-test/512/250GB_total/flow_conv_512.dat"

variables_line, headers, processed_data = parse_tecplot_data(input_file)
save_tecplot_data(output_file, variables_line, headers, processed_data)

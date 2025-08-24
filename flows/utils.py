def fortran_double_str(x: float) -> str:
    """
    Write floating number as fortran style, e.g.

    fortran_double_str(3.0)      => 3.0d0
    fortran_double_str(0.000001) => 1.0d-6
    fortran_double_str(45)       => 4.5d1
    """
    
    # Use scientific notation with one digit before the decimal
    s = f"{x:.15e}"   # gives something like '1.000000000000000e+00'
    
    # Replace 'e' with 'd'
    s = s.replace("e", "d")
    
    # Remove leading '+' in exponent if present
    mantissa, exp = s.split("d")
    exp = exp.lstrip("+").lstrip("0") or "0"  # handle cases like 'd+00'
    
    return f"{float(mantissa)}d{exp}"



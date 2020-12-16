
function [res] = raw2deg(raw,motor)
raw_min_pan = 35; raw_max_pan = 4077;
deg_min_pan = 3; deg_max_pan = 358;
raw_min_tilt = 2595; raw_hor_tilt = 2280; raw_max_tilt = 1595;
deg_min_tilt = 28; deg_hor_tilt = 0; deg_max_tilt = -60.9;
raw_deg = 11.37777; deg_raw = 1 / raw_deg;
if(strcmp(motor,'pan'))
    res = raw*deg_raw;
elseif(strcmp(motor,'tilt'))
    res =[raw - raw_max_tilt]*deg_raw + deg_max_tilt;
end
end
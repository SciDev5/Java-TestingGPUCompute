__kernel void testKernel(__global const float* a, __global const float* b, __global float* out) {
    int GID = get_global_id(0);
    out[GID] = b[GID] * (b[GID]+a[max(GID - 1,0)]) + a[GID] * (a[GID]+b[max(GID - 1, 0)]);
}
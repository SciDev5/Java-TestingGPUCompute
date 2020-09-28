package me.scidev5.testingGPUCompute;

import static org.jocl.CL.*;

import org.jocl.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.Arrays;
import java.util.Iterator;
import java.util.stream.Stream;

public class TestMain {

    public static void main(String[] args) throws FileNotFoundException, URISyntaxException {
        String programSource = readProgramSource();

        int dataLen = 100;
        float[] dataInA = new float[dataLen]; Pointer dataInAPtr = Pointer.to(dataInA);
        float[] dataInB = new float[dataLen]; Pointer dataInBPtr = Pointer.to(dataInB);
        float[] dataOut = new float[dataLen]; Pointer dataOutPtr = Pointer.to(dataOut);

        for (int i = 0; i < dataLen; i++) {
            dataInA[i] = i;
            dataInB[i] = (float)(dataLen/2-i);
        }


        CL.setExceptionsEnabled(true);


        // Get the platform and device information.
        cl_platform_id platform = getPlatform(0);
        cl_device_id device = getDevice(platform,0, CL_DEVICE_TYPE_ALL);

        // Create a gpu compute context.
        cl_context_properties ctxProperties = new cl_context_properties();
        ctxProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
        cl_context context = clCreateContext(ctxProperties, 1, new cl_device_id[] { device }, null, null, null);

        // Setup command queue for handling the order of gpu data requests.
        cl_queue_properties cmdQProperties = new cl_queue_properties();
        cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, device, cmdQProperties, null);

        // Buffer-ify data.
        cl_mem dataInABuff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dataLen * Sizeof.cl_float, dataInAPtr, null);
        cl_mem dataInBBuff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dataLen * Sizeof.cl_float, dataInBPtr, null);
        cl_mem dataOutBuff = clCreateBuffer(context, CL_MEM_READ_WRITE, dataLen * Sizeof.cl_float, dataOutPtr, null);

        // Load and compile program, find kernal.
        cl_program program = createCompileProgram(context, new String[]{ programSource });
        cl_kernel kernel = clCreateKernel(program, "testKernel", null);

        // Evaluate kernal.
        long[] workerDimensions = new long[] { dataLen };
        setKernalArgs(kernel, new cl_mem[]{ dataInABuff, dataInBBuff, dataOutBuff }, 0);
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, workerDimensions, null, 0, null, null);

        // Pull data from buffer.
        clEnqueueReadBuffer(commandQueue, dataOutBuff, CL_TRUE, 0, Sizeof.cl_float * dataLen, dataOutPtr, 0, null, null);

        // Release memory / gpu resources.
        clReleaseMemObject(dataInABuff);
        clReleaseMemObject(dataInBBuff);
        clReleaseMemObject(dataOutBuff);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
        clReleaseDevice(device);

        // Display results and compare to cpu.
        System.out.print("GPU: ");
        System.out.println(Arrays.toString(dataOut));

        // CPU data calculation.
        float[] cpuDataOut = new float[dataLen];
        for (int i = 0; i < dataLen; i++)
            cpuDataOut[i] = dataInB[i] * (dataInB[i] + dataInA[Math.max(i - 1, 0)]) + dataInA[i] * (dataInA[i] + dataInB[Math.max(i - 1, 0)]);

        System.out.print("CPU: ");
        System.out.println(Arrays.toString(cpuDataOut));

        int matched = 0;
        for (int i = 0; i < dataLen; i++)
            if (Math.abs(cpuDataOut[i] - dataOut[i]) < 1.52587891E-5) matched++;

        System.out.println(matched+"/"+dataLen+" were close enough.");
    }

    private static String readProgramSource() throws FileNotFoundException, URISyntaxException {
        final String filePath = "test.cl";

        URL programUri = TestMain.class.getClassLoader().getResource(filePath);
        if (programUri == null) throw new FileNotFoundException(filePath+" was not found.");
        File testProg = new File(programUri.toURI().getPath());
        BufferedReader reader = new BufferedReader(new FileReader(testProg));
        Stream<String> lines = reader.lines();
        StringBuilder programData = new StringBuilder();
        for (Iterator<String> it = lines.iterator(); it.hasNext(); )
            programData.append(it.next()).append("\n");

        return programData.toString();
    }

    private static cl_platform_id getPlatform(int platformIndex) {
        int[] numPlatformsArr = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArr);

        cl_platform_id[] platforms = new cl_platform_id[numPlatformsArr[0]];
        clGetPlatformIDs(platforms.length, platforms, null);
        return platforms[platformIndex];
    }

    private static cl_device_id getDevice(cl_platform_id platform, int deviceIndex, long deviceType) {
        int[] numDevicesArr = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArr);

        cl_device_id[] devices = new cl_device_id[numDevicesArr[0]];
        clGetDeviceIDs(platform, deviceType, devices.length, devices, null);
        return devices[deviceIndex];
    }

    private static cl_program createCompileProgram(cl_context context, String[] programSources) {
        cl_program program = clCreateProgramWithSource(context, programSources.length, programSources, null, null);
        clBuildProgram(program, 0, null, null, null, null);
        return program;
    }

    private static void setKernalArgs(cl_kernel kernel, cl_mem[] buffers, int off) {
        for (int i = 0; i < buffers.length; i++)
            clSetKernelArg(kernel, off + i, Sizeof.cl_mem, Pointer.to(buffers[i]));
    }
}

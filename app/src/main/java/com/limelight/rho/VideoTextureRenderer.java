/*
By Ahmed Hilali

A derivative work based on: http://github.com/izacus/AndroidOpenGLVideoDemo/
See LICENSE.txt
 */

package com.limelight.rho;

import android.content.Context;
import android.graphics.*;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLUtils;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

public class VideoTextureRenderer extends TextureSurfaceRenderer implements SurfaceTexture.OnFrameAvailableListener
{
    private static final String vertexShaderCode =
            "attribute vec4 vPosition;" +
                    "attribute vec4 vTexCoordinate;" +
                    "uniform mat4 textureTransform;" +
                    "varying vec2 v_TexCoordinate;" +
                    "void main() {" +
                    "   v_TexCoordinate = (textureTransform * vTexCoordinate).xy;" +
                    "   gl_Position = vPosition;" +
                    "}";

    private static final String fragmentShaderCode =
            "#extension GL_OES_EGL_image_external : require\n" +
                    "precision highp float;" +
                    "uniform samplerExternalOES texture;" +
                    "uniform float zoomFactor;" +
                    "uniform float distFactor;" +
                    "uniform float wrapEnabled;" +
                    "uniform float singleView;" +
                    // aspect ratios used for correct (non-stretched) SBS 3D rendering
                    "uniform float eyeAspect;" +      // single-eye aspect = (videoWidth/2)/videoHeight
                    "uniform float canvasAspect;" +   // effective eye canvas aspect = (surfaceWidth/2)/surfaceHeight
                    "varying vec2 v_TexCoordinate;" +
                    "void main () {" +
                    "   if(singleView < 0.5) {" +
                    // --- 3D SBS mode (column interleaved for the Leia 4V display) ---
                    "       float localX = v_TexCoordinate.x;" +
                    "       float localY = v_TexCoordinate.y;" +
                    "       float eyeU, eyeV;" +
                    // Contain the single-eye image inside its canvas so it is never stretched.
                    // eyeAspect > canvasAspect  -> eye is wider than canvas -> fit width, letterbox top/bottom
                    // eyeAspect <= canvasAspect -> eye is taller than canvas -> fit height, letterbox left/right
                    "       if (eyeAspect > canvasAspect) {" +
                    "           eyeU = localX;" +
                    "           eyeV = (localY - 0.5) * (eyeAspect / canvasAspect) + 0.5;" +
                    "       } else {" +
                    "           eyeU = (localX - 0.5) * (canvasAspect / eyeAspect) + 0.5;" +
                    "           eyeV = localY;" +
                    "       }" +
                    // Outside the eye image -> render black bars instead of sampling garbage.
                    "       if (eyeU < 0.0 || eyeU > 1.0 || eyeV < 0.0 || eyeV > 1.0) {" +
                    "           gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);" +
                    "       } else {" +
                    "           float viewID = mod(floor(gl_FragCoord.x), 4.0);" +
                    "           float srcX = (viewID < 1.5) ? (eyeU * 0.5) : (eyeU * 0.5 + 0.5);" +
                    "           gl_FragColor = texture2D(texture, vec2(srcX, eyeV));" +
                    "       }" +
                    "   } else {" +
                    // --- 2D / single view mode ---
                    "       gl_FragColor = texture2D(texture, vec2(v_TexCoordinate.x, v_TexCoordinate.y));" +
                    "   }" +
                    "}";

    private static float squareSize = 1.0f;
    private static float squareCoords[] = { -squareSize,  squareSize, 0.0f,   // top left
            -squareSize, -squareSize, 0.0f,   // bottom left
            squareSize, -squareSize, 0.0f,   // bottom right
            squareSize,  squareSize, 0.0f }; // top right

    private static short drawOrder[] = { 0, 1, 2, 0, 2, 3};

    private Context ctx;

    // Texture to be shown in backgrund
    private FloatBuffer textureBuffer;
    private float textureCoords[] = { 0.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
            1.0f, 0.0f, 0.0f, 1.0f,
            1.0f, 1.0f, 0.0f, 1.0f };
    private int[] textures = new int[1];

    private int vertexShaderHandle;
    private int fragmentShaderHandle;
    private int shaderProgram;
    private FloatBuffer vertexBuffer;
    private ShortBuffer drawListBuffer;

    private float zoomFactor = 3.2f;
    private float distortionFactor = 81.0f;
    private float wrapEnabled = 1.0f;
    private float singleView = 0.0f;
    private boolean zoomedIn = false;

    // SBS 3D layout mode (see PreferenceConfiguration.SBS_MODE_*).
    // AUTO (0) = derive the single-eye aspect from the decoded frame.
    // HALF (1) = force half-width SBS: fill the eye canvas (no letterbox).
    // FULL (2) = force full-width SBS: each eye is 16:9, letterboxed to avoid stretching.
    private int sbsMode = 0;

    private SurfaceTexture videoTexture;
    private float[] videoTextureTransform;
    private boolean frameAvailable = false;

    private int videoWidth;
    private int videoHeight;
    private boolean adjustViewport = false;

    public VideoTextureRenderer(Context context, SurfaceTexture texture, int width, int height, OnGlReadyListener listener)
    {
        super(texture, width, height, listener);
        this.ctx = context;
        videoTextureTransform = new float[16];
    }


    private void loadShaders()
    {
        vertexShaderHandle = GLES20.glCreateShader(GLES20.GL_VERTEX_SHADER);
        GLES20.glShaderSource(vertexShaderHandle, vertexShaderCode);
        GLES20.glCompileShader(vertexShaderHandle);
        checkGlError("Vertex shader compile");

        fragmentShaderHandle = GLES20.glCreateShader(GLES20.GL_FRAGMENT_SHADER);
        GLES20.glShaderSource(fragmentShaderHandle, fragmentShaderCode);
        GLES20.glCompileShader(fragmentShaderHandle);
        checkGlError("Pixel shader compile");

        shaderProgram = GLES20.glCreateProgram();
        GLES20.glAttachShader(shaderProgram, vertexShaderHandle);
        GLES20.glAttachShader(shaderProgram, fragmentShaderHandle);
        GLES20.glLinkProgram(shaderProgram);
        checkGlError("Shader program compile");

        int[] status = new int[1];
        GLES20.glGetProgramiv(shaderProgram, GLES20.GL_LINK_STATUS, status, 0);
        if (status[0] != GLES20.GL_TRUE) {
            String error = GLES20.glGetProgramInfoLog(shaderProgram);
            Log.e("SurfaceTest", "Error while linking program:\n" + error);
        }

    }

    public boolean isZoomedIn() {
        return zoomedIn;
    }

    public void setZoomedIn(boolean zoomedIn1) {
        this.zoomedIn = zoomedIn1;
    }

    private void setupVertexBuffer()
    {
        // Draw list buffer
        ByteBuffer dlb = ByteBuffer.allocateDirect(drawOrder. length * 2);
        dlb.order(ByteOrder.nativeOrder());
        drawListBuffer = dlb.asShortBuffer();
        drawListBuffer.put(drawOrder);
        drawListBuffer.position(0);

        // Initialize the texture holder
        ByteBuffer bb = ByteBuffer.allocateDirect(squareCoords.length * 4);
        bb.order(ByteOrder.nativeOrder());

        vertexBuffer = bb.asFloatBuffer();
        vertexBuffer.put(squareCoords);
        vertexBuffer.position(0);
    }


    private void setupTexture(Context context)
    {
        ByteBuffer texturebb = ByteBuffer.allocateDirect(textureCoords.length * 4);
        texturebb.order(ByteOrder.nativeOrder());

        textureBuffer = texturebb.asFloatBuffer();
        textureBuffer.put(textureCoords);
        textureBuffer.position(0);

        // Generate the actual texture
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glGenTextures(1, textures, 0);
        checkGlError("Texture generate");

        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textures[0]);
        checkGlError("Texture bind");

        videoTexture = new SurfaceTexture(textures[0]);
        videoTexture.setOnFrameAvailableListener(this);
    }

    @Override
    protected boolean draw()
    {
        synchronized (this)
        {
            if (frameAvailable)
            {
                videoTexture.updateTexImage();
                videoTexture.getTransformMatrix(videoTextureTransform);
                frameAvailable = false;
            }
            else
            {
                return false;
            }

        }

        // In 3D SBS mode the shader does the aspect correction itself, so we must
        // NOT letterbox the whole quad (that would crop into the eyes). Keep the
        // full surface as the eye canvas instead.
        if (singleView < 0.5f) {
            GLES20.glViewport(0, 0, width, height);
        } else if (adjustViewport) {
            adjustViewport();
        }

        GLES20.glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

        // Draw texture
        GLES20.glUseProgram(shaderProgram);
        int textureParamHandle = GLES20.glGetUniformLocation(shaderProgram, "texture");
        int textureCoordinateHandle = GLES20.glGetAttribLocation(shaderProgram, "vTexCoordinate");
        int positionHandle = GLES20.glGetAttribLocation(shaderProgram, "vPosition");
        int textureTranformHandle = GLES20.glGetUniformLocation(shaderProgram, "textureTransform");
        int zoomHandle = GLES20.glGetUniformLocation(shaderProgram, "zoomFactor");
        int distHandle = GLES20.glGetUniformLocation(shaderProgram, "distFactor");
        int wrapHandle = GLES20.glGetUniformLocation(shaderProgram, "wrapEnabled");
        int singleHandle = GLES20.glGetUniformLocation(shaderProgram, "singleView");
        int eyeAspectHandle = GLES20.glGetUniformLocation(shaderProgram, "eyeAspect");
        int canvasAspectHandle = GLES20.glGetUniformLocation(shaderProgram, "canvasAspect");

        GLES20.glEnableVertexAttribArray(positionHandle);
        GLES20.glVertexAttribPointer(positionHandle, 3, GLES20.GL_FLOAT, false, 4 * 3, vertexBuffer);

        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textures[0]);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glUniform1i(textureParamHandle, 0);

        GLES20.glEnableVertexAttribArray(textureCoordinateHandle);
        GLES20.glVertexAttribPointer(textureCoordinateHandle, 4, GLES20.GL_FLOAT, false, 0, textureBuffer);

        GLES20.glUniformMatrix4fv(textureTranformHandle, 1, false, videoTextureTransform, 0);

        float realZoomFactor = zoomedIn ? (zoomFactor * 1.8f) : zoomFactor;
        GLES20.glUniform1f(zoomHandle, realZoomFactor);
        GLES20.glUniform1f(distHandle, distortionFactor);
        GLES20.glUniform1f(wrapHandle, wrapEnabled);
        GLES20.glUniform1f(singleHandle, singleView);

        // Each eye spans the FULL screen width (column interleaving: every eye samples
        // the left/right half of the frame but is spread across all columns). So the eye
        // canvas aspect is the full surface aspect -- NOT (width/2)/height. Using
        // half-width here is what made "full" get squeezed flat (vertical letterbox).
        float ca = (height > 0) ? ((float) width / (float) height) : 1.0f;

        // Aspect of a single eye as packed in the decoded frame (left/right half).
        float frameEyeAspect = (videoHeight > 0) ? ((videoWidth / 2.0f) / videoHeight) : ca;

        // eyeAspect drives the shader's "contain" fit:
        //  ea == ca -> fill the eye canvas (no letterbox)
        //  ea >  ca -> eye wider than canvas -> letterbox top/bottom
        //  ea <  ca -> eye taller than canvas -> letterbox left/right
        float ea;
        if (sbsMode == 1) {
            // HALF: source is half-SBS (each eye squashed to half width). Fill the
            // canvas so the 2x horizontal unsquash is applied -- this matches the
            // original behavior that looked correct for half-SBS sources.
            ea = ca;
        } else if (sbsMode == 2) {
            // FULL: source is full-SBS; keep each eye's true aspect and contain it
            // (never stretch). Fills exactly when the eye is already 16:9.
            ea = frameEyeAspect;
        } else {
            // AUTO: a squished eye (< 1.0) means half-SBS -> fill; otherwise contain.
            ea = (frameEyeAspect < 1.0f) ? ca : frameEyeAspect;
        }
        GLES20.glUniform1f(eyeAspectHandle, ea);
        GLES20.glUniform1f(canvasAspectHandle, ca);

        GLES20.glDrawElements(GLES20.GL_TRIANGLES, drawOrder.length, GLES20.GL_UNSIGNED_SHORT, drawListBuffer);
        GLES20.glDisableVertexAttribArray(positionHandle);
        GLES20.glDisableVertexAttribArray(textureCoordinateHandle);

        return true;
    }

    public void setZoomFactor(float zoomFactor1) {
        this.zoomFactor = zoomFactor1 / 15.625f;
    }

    public void setDistortionFactor(float distortionFactor1) {
        this.distortionFactor = distortionFactor1;
    }

    public void setWrapEnabled(boolean enabled) {
        this.wrapEnabled = enabled ? 1.0f : 0.0f;
    }

    public void setEnable3D(boolean enabled) {
        this.singleView = enabled ? 0.0f : 1.0f;
    }

    public void setSbsMode(int mode) {
        this.sbsMode = mode;
    }

    private void adjustViewport()
    {
        float surfaceAspect = height / (float)width;
        float videoAspect = videoHeight / (float)videoWidth;

        if (surfaceAspect > videoAspect)
        {
            float heightRatio = height / (float)videoHeight;
            int newWidth = (int)(videoWidth * heightRatio);
            int xOffset = (newWidth - width) / 2;
            GLES20.glViewport(-xOffset, 0, newWidth, height);
        }
        else
        {
            float widthRatio = width / (float)videoWidth;
            int newHeight = (int)(videoHeight * widthRatio);
            int yOffset = (newHeight - height) / 2;
            GLES20.glViewport(0, -yOffset, width, newHeight);
        }

        adjustViewport = false;
    }

    @Override
    protected void initGLComponents()
    {
        setupVertexBuffer();
        setupTexture(ctx);
        loadShaders();
    }

    @Override
    protected void deinitGLComponents()
    {
        GLES20.glDeleteTextures(1, textures, 0);
        GLES20.glDeleteProgram(shaderProgram);
        videoTexture.release();
        videoTexture.setOnFrameAvailableListener(null);
    }

    public void setVideoSize(int width, int height)
    {
        this.videoWidth = width;
        this.videoHeight = height;
        adjustViewport = true;
    }

    public void checkGlError(String op)
    {
        int error;
        while ((error = GLES20.glGetError()) != GLES20.GL_NO_ERROR) {
            Log.e("SurfaceTest", op + ": glError " + GLUtils.getEGLErrorString(error));
        }
    }

    public SurfaceTexture getVideoTexture()
    {
        return videoTexture;
    }

    @Override
    public void onFrameAvailable(SurfaceTexture surfaceTexture)
    {
        synchronized (this)
        {
            frameAvailable = true;
        }
    }
}

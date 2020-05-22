package com.example.tflitepractice;

        import androidx.annotation.ColorInt;
        import androidx.appcompat.app.AppCompatActivity;

        import android.graphics.Bitmap;
        import android.graphics.BitmapFactory;
        import android.os.Bundle;
        import android.util.Log;
        import android.view.View;
        import android.widget.ImageView;
        import android.widget.TextView;

        import org.tensorflow.lite.Interpreter;
        import org.tensorflow.lite.support.common.FileUtil;

        import java.io.IOException;
        import java.io.InputStream;
        import java.nio.ByteBuffer;
        import java.nio.ByteOrder;
        import java.nio.MappedByteBuffer;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = MainActivity.class.getSimpleName();

    private ImageView mImageView;
    private TextView mResultView;
    private int mImageIndex;
    private Interpreter tflite;  // Model
    /*
        Image data inference flow:
        Input image (.png)
        => Bitmap instance
        => ByteBuffer
        => int array
        => inference (model)
        => float array (probabilities)
     */
    private ByteBuffer mImageBuffer;
    private int[] mImagePixels;
    private float[][] mResultBuffer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mImageView = findViewById(R.id.image);
        mResultView = findViewById(R.id.result);

        mImageIndex = -1;

        try {
            MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(this, "mnist.tflite");
            tflite = new Interpreter(tfliteModel);
        } catch (IOException e) {
            Log.d(TAG, e.getMessage());
        }

        mImageBuffer = ByteBuffer.allocateDirect(4 * 28 * 28);
        mImageBuffer.order(ByteOrder.nativeOrder());

        mImagePixels = new int[28 * 28];
        mResultBuffer = new float[1][10];
    }

    public void onButtonClick(View view) {
        Log.d(TAG, "onButtonClick");

        try {
            // Load an image
            InputStream inputStream = getAssets().open("images/" + ++mImageIndex + ".png");
            Bitmap bitmap = BitmapFactory.decodeStream(inputStream);

            // Show the image
            mImageView.setImageBitmap(bitmap);

            // Copy the image to "mImageBuffer"
            bitmap2ImageBuffer(bitmap);

            // Run inference
            tflite.run(mImageBuffer, mResultBuffer);

            // Get and show the inference result
            mResultView.setText("Result: " + parseResult());
        } catch (IOException e) {
            Log.d(TAG, e.getMessage());
        }
    }

    private void bitmap2ImageBuffer(Bitmap bitmap) {
        mImageBuffer.rewind();
        bitmap.getPixels(mImagePixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int color = mImagePixels[pixel++];
                mImageBuffer.putFloat(grayscale(color));
            }
        }
    }

    private int parseResult() {
        int result = -1;
        float prob = 0;
        for (int i = 0; i < 10; i++) {
            if (prob < mResultBuffer[0][i]) {
                result = i;
                prob = mResultBuffer[0][i];
            }
        }
        return result;
    }

    private float grayscale(@ColorInt int color) {
        int A = (color >> 24) & 0xff;
        int R = (color >> 16) & 0xff;
        int G = (color >>  8) & 0xff;
        int B = (color      ) & 0xff;
        return .299f * R + .587f * G + .114f * B;
    }
}

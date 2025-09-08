package com.example.econova

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.zxing.BinaryBitmap
import com.google.zxing.MultiFormatReader
import com.google.zxing.RGBLuminanceSource
import com.google.zxing.common.HybridBinarizer

class MainpageActivity : AppCompatActivity() {

    private val REQUEST_CODE_CAMERA_PERMISSION = 101  // Unique request code for camera permission

    // Registering the ActivityResultLauncher for the camera intent
    private val openCameraLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == RESULT_OK) {
                val imageBitmap = result.data?.extras?.get("data") as? Bitmap // Cast to Bitmap
                imageBitmap?.let {
                    val imageView = findViewById<ImageView>(R.id.imageView) // Your ImageView
                    imageView.setImageBitmap(it) // Set the image in an ImageView
                    // Start QR scan after capturing the image
                    scanQRCode(it)
                }
            } else {
                Toast.makeText(this, "Failed to capture image", Toast.LENGTH_SHORT).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_mainpage)

        // Initializing buttons
        val openSensorDataButton = findViewById<Button>(R.id.opensensordata)
        val openCameraButton = findViewById<Button>(R.id.opencamera)

        // Request Camera permission at runtime if not granted
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
            != android.content.pm.PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(android.Manifest.permission.CAMERA),
                REQUEST_CODE_CAMERA_PERMISSION
            )
        }

        // Camera button click listener
        openCameraButton.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
                == android.content.pm.PackageManager.PERMISSION_GRANTED) {
                // Create an Intent to open the camera
                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                if (cameraIntent.resolveActivity(packageManager) != null) {
                    openCameraLauncher.launch(cameraIntent)  // Launching the camera
                } else {
                    Toast.makeText(this, "No Camera App Found", Toast.LENGTH_SHORT).show()
                }
            } else {
                Toast.makeText(this, "Camera Permission Denied", Toast.LENGTH_SHORT).show()
            }
        }

        // Open sensor data page on button click
        openSensorDataButton.setOnClickListener {
            val intent = Intent(this@MainpageActivity, SensorData::class.java)
            startActivity(intent)
        }

        // Apply the resized drawable to the right of the button
        val factoryDrawable = ContextCompat.getDrawable(this, R.drawable.baseline_factory_24)
        factoryDrawable?.setBounds(0, 0, 80, 80) // Resize the drawable
        val button = findViewById<Button>(R.id.your_button_id)
        button.setCompoundDrawables(null, null, factoryDrawable, null)
    }

    // Method to scan the QR code from the captured image
    private fun scanQRCode(imageBitmap: Bitmap) {
        try {
            val width = imageBitmap.width
            val height = imageBitmap.height
            val pixels = IntArray(width * height)
            imageBitmap.getPixels(pixels, 0, width, 0, 0, width, height)

            val source = RGBLuminanceSource(width, height, pixels)
            val binaryBitmap = BinaryBitmap(HybridBinarizer(source))

            val reader = MultiFormatReader()
            val result = reader.decode(binaryBitmap)

            result?.let {
                val qrCodeData = it.text
                Toast.makeText(this, "Scanned QR Code: $qrCodeData", Toast.LENGTH_SHORT).show()

                // Optionally open the URL if it is a valid URL
                try {
                    val intent = Intent(Intent.ACTION_VIEW, Uri.parse(qrCodeData))
                    startActivity(intent)
                } catch (e: Exception) {
                    Toast.makeText(this, "Invalid QR Code URL", Toast.LENGTH_SHORT).show()
                }
            }
        } catch (e: Exception) {
            Toast.makeText(this, "QR Code scan failed: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    // Handle the result from the permission request
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_CAMERA_PERMISSION) {
            if (grantResults.isNotEmpty() && grantResults[0] == android.content.pm.PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Camera Permission Granted", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Camera Permission Denied", Toast.LENGTH_SHORT).show()
            }
        }
    }
}

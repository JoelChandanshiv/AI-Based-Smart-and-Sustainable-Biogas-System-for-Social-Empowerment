package com.example.econova

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import okhttp3.Call
import okhttp3.Callback
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import org.json.JSONException
import org.json.JSONObject
import java.io.IOException

class SensorData : AppCompatActivity() {

    private lateinit var handler: Handler
    private val updateInterval: Long = 5000  // Fetch data every 5 seconds
    private val okHttpClient = OkHttpClient()
    private lateinit var progressBar: ProgressBar
    private lateinit var textView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.sensor)  // Ensure this points to the correct layout file

        // Find views in the layout
        progressBar = findViewById(R.id.progressBar)
        textView = findViewById(R.id.textview)

        // Initialize the handler
        handler = Handler(Looper.getMainLooper())

        // Start fetching sensor data periodically
        startDataFetch()
    }

    private fun startDataFetch() {
        handler.post(object : Runnable {
            override fun run() {
                fetchSensorData()
                handler.postDelayed(this, updateInterval)  // Repeat every 5 seconds
            }
        })
    }

    private fun fetchSensorData() {
        // Show the ProgressBar while the request is in progress
        progressBar.visibility = View.VISIBLE
        textView.text = "Loading data..."

        // Build the HTTP request to your Flask server
        val request = Request.Builder()
            .url("http://10.10.230.255:5000/latest-sensor-data")  // Replace with the correct server IP and endpoint
            .build()

        // Execute the request asynchronously
        okHttpClient.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                // Handle the failure
                runOnUiThread {
                    progressBar.visibility = View.GONE  // Hide the ProgressBar
                    textView.text = "Network not found: ${e.localizedMessage}"
                    Toast.makeText(this@SensorData, "Network not found", Toast.LENGTH_LONG).show()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    val responseBody = response.body?.string()

                    // Parse the JSON and update the UI
                    runOnUiThread {
                        progressBar.visibility = View.GONE  // Hide the ProgressBar
                        if (responseBody != null) {
                            try {
                                // Parse the JSON response
                                val jsonObject = JSONObject(responseBody)
                                val sensorData = jsonObject.getJSONObject("latest_sensor_data")
                                val predictions = jsonObject.getJSONArray("latest_model_prediction")
                                val status = jsonObject.getString("status")

                                // Extract sensor values
                                val temperature = sensorData.getDouble("temperature")
                                val co2 = sensorData.getDouble("co2")
                                val methane = sensorData.getDouble("methane")
                                val pH = sensorData.getDouble("pH")
                                val pressure = sensorData.getDouble("pressure")

                                // Determine biogas quality
                                val biogasQuality = when {
                                    predictions[0] == 1.0 -> "High"
                                    predictions[1] == 1.0 -> "Low"
                                    predictions[2] == 1.0 -> "Optimal"
                                    else -> "Unknown"
                                }

                                // Format the data for display
                                val formattedData = """
                        Temperature: $temperature °C
                        CO2: $co2 ppm
                        Methane: $methane ppm
                        pH Level: $pH
                        Pressure: $pressure hPa

                        Biogas Quality: $biogasQuality
                    """.trimIndent()

                                // Display the formatted data
                                textView.text = formattedData

                            } catch (e: JSONException) {
                                textView.text = "Error: Invalid JSON format"
                            }
                        } else {
                            textView.text = "Error: Empty Response"
                        }
                    }
                } else {
                    // Handle unsuccessful response
                    runOnUiThread {
                        progressBar.visibility = View.GONE  // Hide the ProgressBar
                        Toast.makeText(this@SensorData, "Request failed", Toast.LENGTH_LONG).show()
                    }
                }
            }
        })
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacksAndMessages(null)  // Stop handler when activity is destroyed
    }
}

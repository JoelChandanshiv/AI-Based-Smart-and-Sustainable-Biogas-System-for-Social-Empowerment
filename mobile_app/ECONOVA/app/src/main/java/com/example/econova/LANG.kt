package com.example.econova

import android.content.Intent
import android.content.SharedPreferences
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import java.util.Locale

class LanguageSelectionActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Set the content view from the XML layout
        setContentView(R.layout.lang)

        // Initialize shared preferences to store selected language
        val sharedPreferences: SharedPreferences = getSharedPreferences("AppPreferences", MODE_PRIVATE)

        // Reference the buttons from the layout
        val btnEnglish = findViewById<Button>(R.id.btn_english)
        val btnHindi = findViewById<Button>(R.id.btn_hindi)
        val btnTamil = findViewById<Button>(R.id.btn_tamil)
        val btnTelugu = findViewById<Button>(R.id.btn_telugu)
        val btnMarathi = findViewById<Button>(R.id.btn_marathi)
        val btnNext = findViewById<Button>(R.id.btn_next)

        // Set up onClick listeners for each button to change the language
        btnEnglish.setOnClickListener { setLanguage("en") }
        btnHindi.setOnClickListener { setLanguage("hi") }
        btnTamil.setOnClickListener { setLanguage("ta") }
        btnTelugu.setOnClickListener { setLanguage("te") }
        btnMarathi.setOnClickListener { setLanguage("mr") }

        // Button to proceed to the main activity
        btnNext.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
            finish()
        }
    }

    private fun setLanguage(languageCode: String) {
        // Save the selected language in SharedPreferences
        val sharedPreferences: SharedPreferences = getSharedPreferences("AppPreferences", MODE_PRIVATE)
        sharedPreferences.edit().putString("SelectedLanguage", languageCode).apply()

        // Change the language for the app
        val locale = Locale(languageCode)
        Locale.setDefault(locale)
        val config = resources.configuration
        config.setLocale(locale)
        resources.updateConfiguration(config, resources.displayMetrics)

        // Reload the current activity to apply the new language
        recreate()
    }
}

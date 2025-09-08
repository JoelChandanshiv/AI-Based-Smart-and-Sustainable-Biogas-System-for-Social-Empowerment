package com.example.econova

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.econova.databinding.ActivityLoginBinding
import com.google.firebase.auth.FirebaseAuth

class LoginActivity : AppCompatActivity() {

    private lateinit var binding: ActivityLoginBinding
    private lateinit var firebaseAuth: FirebaseAuth
    private lateinit var myPreference: MyPreference

    // Added Marathi as a supported language
    private val languageList = arrayOf("English", "हिंदी (Hindi)", "मराठी (Marathi)")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityLoginBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize FirebaseAuth and Preference
        firebaseAuth = FirebaseAuth.getInstance()
        myPreference = MyPreference(this)

        // Set up the spinner with language options
        binding.spinner.adapter = ArrayAdapter(this, android.R.layout.simple_list_item_1, languageList)

        // Get saved language from shared preferences and set the spinner's position
        val savedLanguage = myPreference.getLanguage() // Change to getLanguage() to reflect the proper preference
        val index = languageList.indexOfFirst { it.contains(savedLanguage, ignoreCase = true) }
        if (index >= 0) {
            binding.spinner.setSelection(index)
        }

        // Set up the Login button click listener
        binding.loginbutton.setOnClickListener {
            val email = binding.loginEmail.text.toString().trim()
            val password = binding.loginPassword.text.toString().trim()

            if (email.isNotEmpty() && password.isNotEmpty()) {
                firebaseAuth.signInWithEmailAndPassword(email, password).addOnCompleteListener { task ->
                    if (task.isSuccessful) {
                        val intent = Intent(this, MainpageActivity::class.java)
                        startActivity(intent)
                        finish()
                    } else {
                        Toast.makeText(this, task.exception?.message ?: "Login failed", Toast.LENGTH_SHORT).show()
                    }
                }
            } else {
                Toast.makeText(this, "Fields cannot be empty", Toast.LENGTH_SHORT).show()
            }
        }

        // Set up the Spinner button click listener to save the selected language
        binding.spinnerButton.setOnClickListener {
            val selectedLanguage = languageList[binding.spinner.selectedItemPosition]
            myPreference.setLanguage(getLanguageCode(selectedLanguage)) // Save the language code (like "en", "hi", "mr")
            recreate() // Recreate the activity to apply the language change
        }

        // Set up the redirect to Sign-up page
        binding.loginredirectText.setOnClickListener {
            val signupIntent = Intent(this, MainActivity::class.java)
            startActivity(signupIntent)
        }
    }

    /**
     * This method attaches a base context to ensure that the correct locale is applied
     */
    override fun attachBaseContext(newBase: Context?) {
        super.attachBaseContext(newBase?.let {
            myPreference = MyPreference(it)
            val savedLanguageCode = myPreference.getLanguage() // Changed to getLanguage()
            MyContextWrapper.wrap(it, savedLanguageCode) // Apply the language change
        })
    }

    /**
     * This method returns the language code based on the user's selected language
     */
    private fun getLanguageCode(language: String): String {
        return when {
            language.contains("Hindi", ignoreCase = true) -> "hi"
            language.contains("Marathi", ignoreCase = true) -> "mr"
            else -> "en"
        }
    }
}

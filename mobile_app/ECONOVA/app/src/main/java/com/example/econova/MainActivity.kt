package com.example.econova

import android.content.Intent
import android.os.Bundle
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import com.example.econova.databinding.ActivityMainBinding
import com.google.firebase.auth.FirebaseAuth

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var firebaseAuth: FirebaseAuth

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)

        enableEdgeToEdge()
        setContentView(binding.root)


            firebaseAuth = FirebaseAuth.getInstance()

            binding.signupButton.setOnClickListener {
                val email = binding.signupEmail.text.toString()
                val password = binding.signupPassword.text.toString()
                val confirmPassword = binding.signupConfirm.text.toString()

                if (email.isNotEmpty() && password.isNotEmpty() && confirmPassword.isNotEmpty()){
                    if(password == confirmPassword){
                        firebaseAuth.createUserWithEmailAndPassword(email,password).addOnCompleteListener{
                            if(it.isSuccessful){
                                val intent = Intent(this,LoginActivity::class.java)
                                startActivity(intent)
                            }else{
                                Toast.makeText(this,it.exception.toString(), Toast.LENGTH_SHORT).show()
                            }
                        }
                    }else{
                        Toast.makeText(this, "Password Does not matched", Toast.LENGTH_SHORT).show()
                    }
                }else{
                    Toast.makeText(this, "Fields cannot be empty", Toast.LENGTH_SHORT).show()
                }
            }
            binding.loginredirectText.setOnClickListener {
                val loginIntent = Intent(this, LoginActivity::class.java)
                startActivity(loginIntent)
            }

        }
    }

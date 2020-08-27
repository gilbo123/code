<template>
  <div class="Weights">
    <!-- <img alt="pano" src="./assets/pano_small.jpg"/> -->
    <!-- <First title="Strawberry Dashboard"/> -->
    <!-- <First msg="First Component"/> -->
    <h1>Strawberry Dashboard</h1>
    <h2>Punnet Weights</h2>
    <vue-frappe
      id="weights"
      title="Punnet weights"
      type="line"
      :labels="this.labels"
      :height="500"
      :colors="['#008F68', '#FAE042', '#32a852']"
      :lineOptions="{regionFill: 1}"
      :dataSets="this.defectData">
    </vue-frappe>
  </div>
</template>

<script>

//Firebase
import * as firebase from "firebase/app";
require("firebase/app");
require("firebase/auth")
require('firebase/database')


//var serviceAccount = require("../config/smart_weigh_vue.json")

// Your web app's Firebase configuration
var firebaseConfig = {
    apiKey: "AIzaSyAwK2qlJivRabTefhM9__Mu6Cdlhmfp8tw",
    authDomain: "smart-weight-78680.firebaseapp.com",
    databaseURL: "https://smart-weight-78680.firebaseio.com",
    projectId: "smart-weight-78680",
    storageBucket: "smart-weight-78680.appspot.com",
    messagingSenderId: "482255777827",
    appId: "1:482255777827:web:5cb4ac6e02f05720984196",
    measurementId: "G-H8MVLWWEB4"
  };

// Initialize Firebase
firebase.initializeApp(firebaseConfig);

firebase.auth().signInAnonymously()
.then(function() {
   console.log('Logged in as Anonymous!')
   }).catch(function(error) {
    console.log(error.code);
    console.log(error.message);
});

var date = '2020-08-18'
var ref = firebase.database().ref('punnet_weights').child(date).limitToLast(50)
var scaleVals = []
var scaleLabels = []
ref.on("value", function(snapshot) {
    snapshot.forEach(function(node) {
      scaleVals.push(node.child('weight').val())
      scaleLabels.push(node.child('time').child('hour').val() + ":" + node.child('time').child('minute').val())
    })
    console.log("got data: " + snapshot)
  }, function (error) {
    console.log("Error: " + error.code)
  }
);

console.log(scaleVals)


//graph type
var defectType = 'line'

export default {
  name: 'Weights',
  data() {
    return {
      defectData: [{
                  name: "Weights", chartType: defectType,
                  values: scaleVals
              }
            ],
      labels: scaleLabels,
      // props: {
      //   message: weights.toString
      // }
    }
  },
}



</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>

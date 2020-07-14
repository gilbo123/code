<template>
  <div class="Bar">
    <!-- <img alt="pano" src="./assets/pano_small.jpg"/> -->
    <!-- <First title="Strawberry Dashboard"/> -->
    <!-- <First msg="First Component"/> -->
    <h1>Strawberry Dashboard</h1>
    <h2>Strawberry Defects by Month</h2>
    <vue-frappe
      id="defects"
      title="Number of defects by month"
      type="line"
      :labels="['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']"
      :height="500"
      :colors="['#008F68', '#FAE042', '#32a852', '', '']"
      :lineOptions="{regionFill: 1}"
      :dataSets="this.defectData">
    </vue-frappe>

  </div>
</template>

<script>
// import HelloWorld from './components/HelloWorld.vue'
// import First from './components/First.vue'

var defectType = 'bar'

export default {
  name: 'Bar',
  data() {
    return {
      defectData: [{
                  name: "Underripe", chartType: defectType,
                  values: [40, 275, 370, 455, 572, 580, 585, 387, 288, 90, 2, 0]
              },
              {
                  name: "Overripe", chartType: defectType,
                  values: [8, 85, 91, 89, 85, 190, 297, 410, 412, 321, 30, 5]
              },
              {
                  name: "Bruise", chartType: defectType,
                  values: [0, 5, 291, 289, 285, 290, 297, 310, 312, 321, 33, 4]
              },
              {
                  name: "Foreign", chartType: defectType,
                  values: [0, 15, 51, 89, 85, 90, 97, 50, 12, 1, 0, 0]
              },
              {
                  name: "Small", chartType: defectType,
                  values: [80, 85, 91, 189, 385, 290, 97, 61, 99, 21, 30, 5]
              }
            ],
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

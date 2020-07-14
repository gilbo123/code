<template>
  <div class="Pie">
    <!-- <img alt="pano" src="./assets/pano_small.jpg"/> -->
    <!-- <First msg="First Component"/> -->
    <h2>Total Breakdown</h2>
    <vue-frappe
      id="breakdown"
      title="Total Breakdown"
      type="pie"
      :labels="['Underripe', 'Overripe', 'Bruise', 'Foreign', 'Small']"
      :height="500"
      :colors="['#008F68', '#FAE042', '#32a852', '', '']"
      :dataSets="this.breakdownData">
    </vue-frappe>
  </div>
</template>


<script>
var breakdownType = 'pie'
export default {
  name: 'Pie',
  data() {
    return {
            breakdownData: [{
                        name: "Defects", chartType: breakdownType,
                        values: [3644, 2023, 2437, 490, 1433]
                        }
                      ]

    }
  }
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
